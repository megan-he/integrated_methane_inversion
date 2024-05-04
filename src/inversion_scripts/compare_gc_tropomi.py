import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import glob
import re
from utils import load_obj, plot_field

# Get observed and GEOS-Chem-simulated TROPOMI columns
def aggregate_data(data_dir, data_posterior):
    
    files = np.sort(os.listdir(data_dir))
    lat = np.array([])
    lon = np.array([])
    tropomi = np.array([])
    geos_prior = np.array([])
    geos_posterior = np.array([])
    observation_count = np.array([])

    for f in files:
        # Get paths
        if ".pkl" in f:
            pth = os.path.join(data_dir, f)
            pth_posterior = os.path.join(data_posterior, f)
            # Load TROPOMI/GEOS-Chem and Jacobian matrix data from the .pkl file
            obj = load_obj(pth)
            obj_posterior = load_obj(pth_posterior)
            # If there aren't any TROPOMI observations on this day, skip
            if obj["obs_GC"].shape[0] == 0:
                continue
            # Otherwise, grab the TROPOMI/GEOS-Chem data
            obs_GC = obj["obs_GC"]
            obs_GC_posterior = obj_posterior["obs_GC"]
            # Only consider data within latitude and longitude bounds
            ind = np.where(
                (obs_GC[:, 2] >= lon_bounds[0])
                & (obs_GC[:, 2] <= lon_bounds[1])
                & (obs_GC[:, 3] >= lat_bounds[0])
                & (obs_GC[:, 3] <= lat_bounds[1])
            )
            if len(ind[0]) == 0:  # Skip if no data in bounds
                continue
            if obs_GC.shape[0] != obs_GC_posterior.shape[0]:
                continue
            obs_GC = obs_GC[ind[0], :]  # TROPOMI and GEOS-Chem data within bounds
            obs_GC_posterior = obs_GC_posterior[ind[0], :]
            # Record lat, lon, tropomi ch4, and geos ch4
            lat = np.concatenate((lat, obs_GC[:, 3]))
            lon = np.concatenate((lon, obs_GC[:, 2]))
            tropomi = np.concatenate((tropomi, obs_GC[:, 0]))
            geos_prior = np.concatenate((geos_prior, obs_GC[:, 1]))
            observation_count = np.concatenate((observation_count, obs_GC[:, 4]))
            geos_posterior = np.concatenate((geos_posterior, obs_GC_posterior[:, 1]))

    df = pd.DataFrame()
    df["lat"] = lat
    df["lon"] = lon
    df["tropomi"] = tropomi
    df["geos_prior"] = geos_prior
    df["geos_posterior"] = geos_posterior
    df["diff_tropomi_prior"] = geos_prior - tropomi
    df["diff_tropomi_posterior"] = geos_posterior - tropomi
    df["observation_count"] = observation_count

    return df


def plot_comparison(ds, lat_bounds, lon_bounds):

    # Open the state vector file
    state_vector_filepath = "../StateVector.nc"
    state_vector = xr.load_dataset(state_vector_filepath)
    state_vector_labels = state_vector["StateVector"]

    # Identify the last element of the region of interest
    last_ROI_element = int(np.nanmax(state_vector_labels.values))

    # Define mask for region of interest
    mask = (state_vector_labels <= last_ROI_element)

    # Mean TROPOMI XCH4 columns on 0.1 x 0.1 grid
    fig = plt.figure(figsize=(8, 8))
    plt.rcParams.update({"font.size": 16})
    ax = fig.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})

    plot_field(
        ax,
        ds["tropomi"],
        cmap="Spectral_r",
        vmin=1750,
        vmax=1950,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        title="TROPOMI $X_{CH4}$",
        cbar_label="Column mixing ratio (ppb)",
        mask=mask,
        only_ROI=False,
    )
    plt.savefig(
        "tropomi",
        bbox_inches="tight",
        dpi=150,
    )

    # Plot differences between GEOS-Chem prior simulation and TROPOMI XCH4 on 2 x 2.5 grid
    fig = plt.figure(figsize=(8, 8))
    plt.rcParams.update({"font.size": 16})
    ax = fig.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})

    plot_field(
        ax,
        ds["diff_tropomi_prior"],
        cmap="RdBu_r",
        vmin=-40,
        vmax=40,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        title="GEOS-Chem Prior $-$ TROPOMI",
        cbar_label="ppb",
        mask=mask,
        only_ROI=False,
    )
    plt.savefig(
        "diff_prior_tropomi",
        bbox_inches="tight",
        dpi=150,
    )

    # Plot differences between GEOS-Chem posterior simulation and TROPOMI XCH4 on 2 x 2.5 grid
    fig = plt.figure(figsize=(8, 8))
    plt.rcParams.update({"font.size": 16})
    ax = fig.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})

    plot_field(
        ax,
        ds["diff_tropomi_posterior"],
        cmap="RdBu_r",
        vmin=-40,
        vmax=40,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        title="GEOS-Chem Posterior $-$ TROPOMI",
        cbar_label="ppb",
        mask=mask,
        only_ROI=False,
    )
    plt.savefig(
        "diff_posterior_tropomi",
        bbox_inches="tight",
        dpi=150,
    )

if __name__ == "__main__":
    
    satdat_dir = "./data_converted"
    posterior_dir = "./data_converted_posterior"
    visualization_dir = "./data_visualization"
    posterior_viz_dir = "./data_visualization_posterior"

    # Set up map bounds for plotting
    lat_bounds = [-56.0, 76.0]
    lon_bounds = [-170.0, 167.5]

    superobs_df = aggregate_data(satdat_dir, posterior_dir)
    visualization_df = aggregate_data(visualization_dir, posterior_viz_dir)
    n_obs = len(superobs_df["tropomi"])

    print(
        f'Found {n_obs} super-observations in the domain, representing {np.sum(superobs_df["observation_count"]).round(0)} TROPOMI observations.'
    )
    print(superobs_df.head())


    # Print some error statistics
    print(
        "Bias in prior     :",
        np.round(
            np.average(
                superobs_df["diff_tropomi_prior"], weights=superobs_df["observation_count"]
            ),
            2,
        ),
        "ppb",
    )
    print(
        "Bias in posterior :",
        np.round(
            np.average(
                superobs_df["diff_tropomi_posterior"],
                weights=superobs_df["observation_count"],
            ),
            2,
        ),
        "ppb",
    )
    print(
        "RMSE prior        :",
        np.round(
            np.sqrt(
                np.average(
                    superobs_df["diff_tropomi_prior"] ** 2,
                    weights=superobs_df["observation_count"],
                )
            ),
            2,
        ),
        "ppb",
    )
    print(
        "RMSE posterior    :",
        np.round(
            np.sqrt(
                np.average(
                    superobs_df["diff_tropomi_posterior"] ** 2,
                    weights=superobs_df["observation_count"],
                )
            ),
            2,
        ),
        "ppb",
    )

    # Simple averaging scheme to grid the XCH4 data at 2 x 2.5 resolution
    df_copy = visualization_df.copy()  # save for later
    visualization_df["lat"] = np.round(visualization_df["lat"]/2.0)*2.0
    visualization_df["lon"] = np.round(visualization_df["lon"]/2.0)*2.0
    visualization_df = visualization_df.groupby(["lat", "lon"]).mean()
    ds = visualization_df.to_xarray()

    # Plot and save figures
    plot_comparison(ds, lat_bounds, lon_bounds)
    print("Done plotting")
    