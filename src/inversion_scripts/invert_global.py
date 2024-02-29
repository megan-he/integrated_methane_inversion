#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import numpy as np
from netCDF4 import Dataset
from utils import load_obj, calculate_superobservation_error


def do_inversion(
    n_elements,
    jacobian_dir,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    prior_err=0.5,
    obs_err=15,
    gamma=0.3,
    res="0.25x0.3125",
    jacobian_sf=None,
    perturb_oh=1.2,
    prior_err_bc=0.0,
    prior_err_oh=0.1,
):
    """
    After running jacobian.py, use this script to perform the inversion and save out results.

    Arguments
        n_elements   [int]   : Number of state vector elements
        jacobian_dir [str]   : Directory where the data from jacobian.py are stored
        lon_min      [float] : Minimum longitude
        lon_max      [float] : Maximum longitude
        lat_min      [float] : Minimum latitude
        lat_max      [float] : Maximum latitude
        prior_err    [float] : Prior error standard deviation (default 0.5)
        obs_err      [float] : Observational error standard deviation (default 15 ppb)
        gamma        [float] : Regularization parameter (default 0.25)
        res          [str]   : Resolution string from config.yml (default '0.25x0.3125')
        jacobian_sf  [str]   : Path to Jacobian scale factors file if using precomputed K
        prior_err_bc [float] : Prior error standard deviation (default 0.0)
        prior_err_oh [float] : Prior error standard deviation (default 0.0)

    Returns
        xhat         [float] : Posterior scaling factors
        ratio        [float] : Change from prior     [xhat = 1 + ratio]
        KTinvSoK     [float] : K^T*inv(S_o)*K        [part of inversion equation]
        KTinvSoyKxA  [float] : K^T*inv(S_o)*(y-K*xA) [part of inversion equation]
        S_post       [float] : Posterior error covariance matrix
        A            [float] : Averaging kernel matrix

    """

    # Need to ignore data in the GEOS-Chem 3 3 3 3 buffer zone
    # Shave off one or two degrees of latitude/longitude from each side of the domain
    # ~1 degree if 0.25x0.3125 resolution, ~2 degrees if 0.5x0.6125 resolution
    # This assumes 0.25x0.3125 and 0.5x0.625 simulations are always regional
    if "0.25x0.3125" in res:
        degx = 4 * 0.3125
        degy = 4 * 0.25
    elif "0.5x0.625" in res:
        degx = 4 * 0.625
        degy = 4 * 0.5
    else:
        degx = 0
        degy = 0

    xlim = [lon_min + degx, lon_max - degx]
    ylim = [lat_min + degy, lat_max - degy]

    # Read output data from jacobian.py (virtual & true TROPOMI columns, Jacobian matrix)
    sensi_save_pth = os.path.expanduser("~/mhe/calc_sensi_test")
    files = glob.glob(f"{sensi_save_pth}/sensi_20190101_*.nc")
    files.sort()

    # ==========================================================================================
    # Now we will assemble two different expressions needed for the analytical inversion.
    #
    # These expressions are from eq. (5) and (6) in Zhang et al. (2018) ACP:
    # "Monitoring global OH concentrations using satellite observations of atmospheric methane".
    #
    # Specifically, we are going to solve:
    #   xhat = xA + G*(y-K*xA)
    #        = xA + inv(gamma * K^T*inv(S_o)*K + inv(S_a)) * gamma * K^T*inv(S_o) * (y-K*xA)
    #                          (--------------)                     (-----------------------)
    #                            Expression 1                             Expression 2
    #
    # Expression 1 = "KTinvSoK"
    # Expression 2 = "KTinvSoyKxA"
    #
    # In the code below this becomes
    #   xhat = xA + inv(gamma*KTinvSoK + inv(S_a)) * gamma*KTinvSoyKxA
    #        = xA + ratio
    #        = 1  + ratio      [since xA=1 when optimizing scale factors]
    #
    # We build KTinvSoK and KTinvSoyKxA "piece by piece", loading one jacobian .pkl file at a
    # time. This is so that we don't need to assemble or invert the full Jacobian matrix, which
    # can be very large.
    # ==========================================================================================

    # Initialize two expressions from the inversion equation
    KTinvSoK = np.zeros(
        [n_elements + 1, n_elements + 1], dtype=float
    )  # expression 1: K^T * inv(S_o) * K
    KTinvSoyKxA = np.zeros(
        [n_elements + 1], dtype=float
    )  # expression 2: K^T * inv(S_o) * (y-K*xA)

    tropomi = np.array([])
    geos_prior = np.array([])
    So = np.array([])
    # For each .pkl file generated by jacobian.py:
    # for fi in files:
    for i, fi in enumerate(files):
        print(i, fi)

        # Load TROPOMI/GEOS-Chem and Jacobian matrix data from the .pkl file
        dat = load_obj(fi)

        # Skip if there aren't any TROPOMI observations on this day
        if dat["obs_GC"].shape[0] == 0:
            continue

        # Otherwise, grab the TROPOMI/GEOS-Chem data
        obs_GC = dat["obs_GC"]

        # Only consider data within the new latitude and longitude bounds
        ind = np.where(
            (obs_GC[:, 2] >= xlim[0])
            & (obs_GC[:, 2] <= xlim[1])
            & (obs_GC[:, 3] >= ylim[0])
            & (obs_GC[:, 3] <= ylim[1])
        )

        # Skip if no data in bounds
        if (len(ind[0]) == 0):
            continue

        # TROPOMI and GEOS-Chem data within bounds
        obs_GC = obs_GC[ind[0], :]

        tropomi = np.concatenate((tropomi, obs_GC[:,0]))
        geos_prior = np.concatenate((geos_prior, obs_GC[:,1]))

        # Build Jacobian matrix one column at a time
        if i == 0:
            K = 1e9 * dat["K"][ind[0]]
        else:
            K = np.append(K, 1e9 * dat["K"][ind[0]], axis = 0)

        # weight obs_err based on the observation count to prevent overfitting
        # Note: weighting function defined by Zichong Chen for his
        # middle east inversions. May need to be tuned based on region.
        # From Chen et al. 2023:
        # "Satellite quantification of methane emissions and oil/gas methane
        # intensities from individual countries in the Middle East and North
        # Africa: implications for climate action"
        s_superO_1 = calculate_superobservation_error(obs_err, 1)
        s_superO_p = np.array(
            [
                calculate_superobservation_error(obs_err, p) if p >= 1 else s_superO_1
                for p in obs_GC[:, 4]
            ]
        )
        gP = s_superO_p**2 / s_superO_1**2
        obs_error = gP * obs_err
        # these all have the same shape and are a 1D array

        # check to make sure obs_err isn't negative, set 1 as default value
        obs_error = [obs if obs > 0 else 1 for obs in obs_error]

        # Number of observations
        print("Sum of Jacobian entries:", np.sum(K))
        print("Shape of K: ", K.shape)

        # Apply scaling matrix if using precomputed Jacobian
        if jacobian_sf is not None:
            scale_factors = np.load(jacobian_sf)
            reps = K.shape[0]
            scaling_matrix = np.tile(scale_factors, (reps, 1))
            K *= scaling_matrix

        # Define observational errors (diagonal entries of S_o matrix)
        obs_error = np.power(obs_error, 2)
        So = np.concatenate((So, obs_error))
        
        # If there are any nans in the data, abort
        if (
            # np.any(np.isnan(delta_y))
            np.any(np.isnan(K))
            or np.any(np.isnan(obs_error))
        ):
            print("missing values", fi)
            break

    print(f'K shape: {K.shape}')
    print(f'first column of K: {K[:,0]}')
    print(f'last column of K: {K[:, -1]}')

    # Inverse of prior error covariance matrix, inv(S_a)
    Sa_diag = np.zeros(n_elements + 1)
    Sa_diag.fill(prior_err**2)

    # If optimizing OH, adjust for it in the inversion
    if prior_err_oh > 0.0:
        # add prior error for OH as the last element of the diagonal
        Sa_diag[-1:] = (1/1000)*prior_err_oh**2 # try this
        OH_idx = n_elements # index 1000

    # If optimizing boundary conditions, adjust for it in the inversion
    if prior_err_bc > 0.0:
        bc_idx = n_elements + 1
        # add prior error for BCs as the last 4 elements of the diagonal
        if prior_err_oh > 0.0:
            Sa_diag[-5:] = prior_err_bc**2
            bc_idx -= 5
        else:
            Sa_diag[-4:] = prior_err_bc**2
            bc_idx -= 4

    inv_Sa = np.diag(1 / Sa_diag)  # Inverse of prior error covariance matrix
    
    # Define KTinvSo = K^T * inv(S_o)
    KT = K.transpose()
    KTinvSo = np.zeros(KT.shape, dtype=float)
    for k in range(KT.shape[1]):
        KTinvSo[:, k] = KT[:, k] / So[k]
    KTinvSoK = KTinvSo @ K # expression 1: K^T * inv(S_o) * K
    obs_tropomi = np.asmatrix(tropomi)
    gc_ch4_prior = np.asmatrix(geos_prior)
    delta_y = obs_tropomi.T - gc_ch4_prior.T # y - K*xA [ppb]
    KTinvSoyKxA = (KTinvSo @ delta_y)  # expression 2: K^T * inv(S_o) * (y-K*xA)

    # Solve for posterior scale factors xhat
    term1 = np.linalg.inv(gamma * KTinvSoK + inv_Sa)
    term2 = gamma * KTinvSoyKxA
    ratio = np.linalg.inv(gamma * KTinvSoK + inv_Sa) @ (gamma * KTinvSoyKxA)
    print(f'term1: {term1}')
    print(f'term2: {term2}')
    print(f'ratio: {ratio}')
    
    # update scale factors by 1 to match what geoschem expects
    # Note: if optimizing BCs, the last 4 elements are in concentration 
    # space, so we do not need to add 1
    # xhat = 1 + ratio
    xhat = ratio.copy()
    xhat[:OH_idx] += 1
    # update OH scale factor by perturbation
    if prior_err_oh > 0.0:
        xhat[OH_idx] += perturb_oh
    # np.save('xhat', xhat)

    # Posterior error covariance matrix
    S_post = np.linalg.inv(gamma * KTinvSoK + inv_Sa)

    # Averaging kernel matrix
    A = np.identity(n_elements + 1) - S_post @ inv_Sa
    
    print(f'xhat = {xhat}')
    # Print some statistics
    print("Min:", xhat[:OH_idx].min(), "Mean:", xhat[:OH_idx].mean(), "Max", xhat[:OH_idx].max())

    # Print OH
    print(f"xhat[OH] = {xhat[OH_idx]}")

    return xhat, ratio, KTinvSoK, KTinvSoyKxA, S_post, A


if __name__ == "__main__":
    import sys

    n_elements = int(sys.argv[1])
    jacobian_dir = sys.argv[2]
    output_path = sys.argv[3]
    lon_min = float(sys.argv[4])
    lon_max = float(sys.argv[5])
    lat_min = float(sys.argv[6])
    lat_max = float(sys.argv[7])
    prior_err = float(sys.argv[8])
    obs_err = float(sys.argv[9])
    gamma = float(sys.argv[10])
    res = sys.argv[11]
    jacobian_sf = sys.argv[12]
    perturb_oh = float(sys.argv[13])
    prior_err_BC = float(sys.argv[14])
    prior_err_OH = float(sys.argv[15])

    # Reformat Jacobian scale factor input
    if jacobian_sf == "None":
        jacobian_sf = None

    # Run the inversion code
    out = do_inversion(
        n_elements,
        jacobian_dir,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        prior_err,
        obs_err,
        gamma,
        res,
        jacobian_sf,
        perturb_oh,
        prior_err_BC,
        prior_err_OH,
    )
    xhat = out[0]
    ratio = out[1]
    KTinvSoK = out[2]
    KTinvSoyKxA = out[3]
    S_post = out[4]
    A = out[5]

    # Save results
    dataset = Dataset(output_path, "w", format="NETCDF4_CLASSIC")
    nvar = dataset.createDimension("nvar", n_elements + 1)
    nc_KTinvSoK = dataset.createVariable("KTinvSoK", np.float32, ("nvar", "nvar"))
    nc_KTinvSoyKxA = dataset.createVariable("KTinvSoyKxA", np.float32, ("nvar"))
    nc_ratio = dataset.createVariable("ratio", np.float32, ("nvar"))
    nc_xhat = dataset.createVariable("xhat", np.float32, ("nvar"))
    nc_S_post = dataset.createVariable("S_post", np.float32, ("nvar", "nvar"))
    nc_A = dataset.createVariable("A", np.float32, ("nvar", "nvar"))
    nc_KTinvSoK[:, :] = KTinvSoK
    nc_KTinvSoyKxA[:] = KTinvSoyKxA
    nc_ratio[:] = ratio
    nc_xhat[:] = xhat
    nc_S_post[:, :] = S_post
    nc_A[:, :] = A
    dataset.close()

    print(f"Saved results to {output_path}")
