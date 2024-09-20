import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.inversion_scripts.utils import filter_prior_files, get_mean_emissions

def clusters_2d_to_1d(clusters, data, fill_value=0):
    '''
    Flattens data on the GEOS-Chem grid, and ensures the resulting order is
    ascending with respect to cluster number. Adapted from Hannah Nesser.
    Parameters:
        clusters (xr.Datarray) : 2d array of cluster values for each gridcell.
                                 You can get this directly from a cluster file
                                 used in an analytical inversion.
                                 Dimensions: ('lat','lon')
        data (xr.DataArray)    : Data on a 2d GEOS-Chem grid.
                                 Dimensions: ('lat','lon')
   '''
    # Data must be a dataarray
    assert type(data) == xr.core.dataarray.DataArray, \
           "Input data must be a dataarray."

    # Combine clusters and data into one dataarray
    data = data.to_dataset(name='data')
    data['clusters'] = clusters

    # Convert to a dataframe and reset index to remove lat/lon/time
    # dimensions
    data = data.to_dataframe().reset_index()[['data', 'clusters']]

    # Remove non-cluster datapoints
    data = data[data['clusters'] > 0]

    # Fill nans that may result from data and clusters being different
    # shapes
    data = data.fillna(fill_value)

    # Sort
    data = data.sort_values(by='clusters')

    return data['data'].values

def sectoral_matrix(statevector, emissions):
    '''
    Group emissions by sector and generate W matrix
    '''
    sectors = ['Reservoirs', 'Wetlands', 'BiomassBurn', 'OtherAnth', 'Rice', 
               'Wastewater', 'Landfills', 'Livestock', 'Coal', 'Gas', 'Oil']
    W = pd.DataFrame(columns=sectors)
    for s in sectors:
        emis = emissions['EmisCH4_'+s].squeeze()
        emis *= emissions['AREA']

        emis = clusters_2d_to_1d(statevector, emis)

        W[s] = emis

    return W

def source_attribution(w, xhat, shat=None, a=None):
    '''
    This takes a W matrix and returns xhat, shat, and a in the 
    original units they were provided to the function. Adapted from Hannah Nesser.
    '''

    # Check for singular matrix conditions
    if np.any(w.sum(axis=1) == 0):
        print('Dropping ', w.index[w.sum(axis=1) == 0].values)
        w = w.drop(index=w.index[w.sum(axis=1) == 0].values)

    # Normalize W
    w_tot = w.sum(axis=1)
    w = w/w_tot.values[:, None]

    # xhat red = W xhat
    xhat_red = w @ xhat

    if (shat is not None) and (a is not None):
        # Convert error covariance matrices
        # S red = W S W^T
        shat_red = w @ shat @ w.T

        # Calculate Pearson's correlation coefficient (cov(X,Y)/stdx*stdy)
        stdev_red = np.sqrt(np.diagonal(shat_red)) 
        r_red = shat_red/(stdev_red[:, None]*stdev_red[None, :])

        # Calculate reduced averaging kernel
        # a_red = np.identity(w.shape[0]) - shat_red @ inv(sa_red)
        a_red = w @ a @ w.T @ np.linalg.inv((w @ w.T))
        # return xhat_red, shat_red, a_red
        return xhat_red, shat_red, r_red, a_red
    else:
        return xhat_red

def analyze_OH(data_dir):
    inversion_result = xr.load_dataset(f'{data_dir}/inversion/inversion_result.nc')
    # Keep only OH elements for analysis
    xhat_OH = inversion_result["xhat"][-2:].to_numpy()
    S_post_OH = inversion_result["S_post"].isel(nvar1=slice(-2,None), nvar2=slice(-2,None)).to_numpy()
    A_OH = inversion_result["A"].isel(nvar1=slice(-2,None), nvar2=slice(-2,None)).to_numpy()

    print(S_post_OH)
    print(A_OH)
    print(np.trace(A_OH))


def plot_correlation(w_matrix, kf=False):
    '''Plot error correlation matrix given W'''

    w = w_matrix.T
    inversion_result = xr.load_dataset(f'{data_dir}/inversion/inversion_result.nc')
    # Keep only emission elements for analysis
    xhat_emissions = inversion_result["xhat"][:-2]
    S_post = inversion_result["S_post"].isel(nvar1=slice(None,-2), nvar2=slice(None,-2))
    A = inversion_result["A"].isel(nvar1=slice(None,-2), nvar2=slice(None,-2))

    xhat_reduced, S_post_red, pearson, A_red = source_attribution(w, xhat_emissions, S_post, A)
    cols = pearson.columns.tolist()

    # Plot posterior error correlation matrix
    fig = plt.figure(figsize=(19,15))
    ax = fig.add_subplot(111)
    cax = ax.matshow(pearson.corr(), origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticklabels(cols, rotation=45, fontsize=18)
    ax.set_yticklabels(cols, fontsize=18)
    cbar = fig.colorbar(cax)
    cbar.ax.tick_params(labelsize=20)
    if kf:
        plt.savefig(f'error_corr_{year}{i+5:02d}.png')
    else:
        plt.savefig(f'error_corr_{year}.png')

if __name__ == "__main__":

    year = 2019
    kalman_mode = False
    start_date = "20190101"
    end_date = "20200101"

    data_dir = f"/n/holyscratch01/jacob_lab/mhe/Global_{year}_annual"
    months = [i for i in range(1, 13)]
    emis_files = [f'{data_dir}/hemco_prior_emis/OutputDir/HEMCO_sa_diagnostics.{year}{m:02d}010000.nc'
                for m in months] # list of emissions for first day in each month
    prior_cache_path = f'{data_dir}/hemco_prior_emis/OutputDir/'
    sv_path = f"{data_dir}/StateVector.nc"
    sv = xr.open_dataset(sv_path)
    sv = sv.to_dataarray('StateVector')

    if kalman_mode:
        for i, emis in enumerate(emis_files):
            ds = xr.load_dataset(emis)
            # Save W matrix for each month
            w = sectoral_matrix(sv, ds)

            # print(w.sum()*86400*31*1e-9) # Tg/yr
            print(f"Monthly total for {year}{i+5:02d}: {w.to_numpy().sum()*86400*31*1e-9} Tg")
            w.to_csv(f'{data_dir}/w_{year}{i+5:02d}.csv', index=False)
            plot_correlation(w, kalman_mode)

    else:
        ds = get_mean_emissions(start_date, end_date, prior_cache_path)
        ds.to_netcdf(f"{data_dir}/HEMCO_diagnostics.{year}.nc")
        # Also check annual mean emissions
        emissions_in_kg_per_s = ds["EmisCH4_Total"] * ds["AREA"]
        total = emissions_in_kg_per_s.sum() * 86400 * 365 * 1e-9
        print(f"{total.values:.2f} Tg/yr")

        # Save annual mean W matrix
        w = sectoral_matrix(sv, ds)
        w_total = w.to_numpy().sum() * 86400 * 365 * 1e-9
        print(f"{w_total:.2f} Tg/yr")
        w.to_csv(f'{data_dir}/w_{year}_annual.csv', index=False)

        # Plot
        plot_correlation(w, kalman_mode)

        # Print OH statistics
        analyze_OH(data_dir)