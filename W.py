import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":

    import sys
    make_W = sys.argv[1].lower() == 'true'
    plot_correlation = sys.argv[2].lower() == 'true'

    data_dir = "/n/holyscratch01/jacob_lab/mhe/Global_2019_fast"
    year = 2018
    months = [i for i in range(5, 13)]
    priors = [f'{data_dir}/prior_run/OutputDir/HEMCO_sa_diagnostics.{year}{m:02d}010000.nc'
                 for m in months] # list of emissions for first day in each month
    sv_path = f"{data_dir}/StateVector.nc"
    sv = xr.open_dataset(sv_path)
    sv = sv.to_dataarray('StateVector')

    if make_W:
        # Save W matrix for each month
        for i, prior in enumerate(priors):
            ds = xr.load_dataset(prior)
            
            w = sectoral_matrix(sv, ds)

            # print(w.sum()*86400*31*1e-9) # Tg/yr
            print(f"Monthly total for {year}{i+5:02d}: {w.to_numpy().sum()*86400*31*1e-9} Tg")
            w.to_csv(f'{data_dir}/w_{year}{i+5:02d}.csv', index=False)

    if plot_correlation:
        # Load W matrix
        for i, prior in enumerate(priors):
            w = pd.read_csv(f'{data_dir}/w_{year}{i+5:02d}.csv')
            # anth_cols = ['Reservoirs', 'BiomassBurn', 'OtherAnth', 'Rice', 
            #            'Wastewater', 'Landfills', 'Livestock', 'Coal', 'Gas', 'Oil']
            # bio_cols = ['Wetlands']
            # w['total'] = w.sum(axis=1)
            # w['total_anth'] = w[anth_cols].sum(axis=1)
            # w['total_bio'] = w[bio_cols].sum(axis=1)
            w = w.T
            inversion_result = xr.load_dataset(f'{data_dir}/kf_inversions/period{i+1}/inversion_result.nc')
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
            plt.savefig(f'error_corr_{year}{i+5:02d}.png')
