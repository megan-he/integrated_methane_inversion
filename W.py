import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shapefile
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from src.inversion_scripts.utils import get_mean_emissions

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


def grid_shape_overlap(clusters, x, y, name=None):
    '''
    Make mask of fractional overlaps for grid cells in statevector that intersect with polygon of region boundaries.
    Adapted from Hannah Nesser.
    '''
    # Initialize mask
    mask = np.zeros(int(np.nanmax(clusters.values)))

    # Make a polygon
    c_poly = Polygon(np.column_stack((x, y)))
    if not c_poly.is_valid:
        print(f'Buffering {name}')
        c_poly = c_poly.buffer(0)

    # Get maximum latitude and longitude limits
    lon_max = np.max(x)
    lon_max = 177.5 if lon_max > 177.5 else lon_max # Hardcoded to avoid out of bounds. May not be needed for future shapefiles
    lat_lims = (np.min(y), np.max(y))
    lon_lims = (np.min(x), lon_max)
    
    # Convert that to the GC grid (admittedly using grid cell centers 
    # instead of edges, but that should be consesrvative)
    c_lat_lims = (clusters.lat.values[clusters.lat < lat_lims[0]][-1],
                  clusters.lat.values[clusters.lat > lat_lims[1]][0])
    c_lon_lims = (clusters.lon.values[clusters.lon <= lon_lims[0]][-1],
                  clusters.lon.values[clusters.lon >= lon_lims[1]][0])
    c_clusters = clusters.sel(lat=slice(*c_lat_lims), 
                              lon=slice(*c_lon_lims))
    c_cluster_list = c_clusters.values.flatten()
    c_cluster_list = c_cluster_list[c_cluster_list > 0]

    # Iterate through overlapping grid cells
    for i, gc in enumerate(c_cluster_list):
        # Get center of grid box
        gc_center = c_clusters.where(c_clusters == gc, drop=True)
        gc_center = (gc_center.lon.values[0], gc_center.lat.values[0])
        
        # Get corners
        gc_corners_lon = [gc_center[0] - 2.5/2,
                          gc_center[0] + 2.5/2,
                          gc_center[0] + 2.5/2,
                          gc_center[0] - 2.5/2]
        gc_corners_lat = [gc_center[1] - 2/2,
                          gc_center[1] - 2/2,
                          gc_center[1] + 2/2,
                          gc_center[1] + 2/2]

        # Make polygon
        gc_poly = Polygon(np.column_stack((gc_corners_lon, gc_corners_lat)))

        if gc_poly.intersects(c_poly):
            # Get area of overlap area and GC cell and calculate
            # the fractional contribution of the overlap area
            overlap_area = c_poly.intersection(gc_poly).area
            gc_area = gc_poly.area
            mask[int(gc) - 1] = overlap_area/gc_area

    return mask


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

def regional_matrix(statevector, emissions, regions, w_mask):
    '''
    Group emissions by custom region/continent and generate W matrix
    '''
    for j, shape in enumerate(regions.shapeRecords()):
        if shape.record[1] in w_mask.columns:
            # Add a row to w_state
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            # populate with fractional overlaps for grid cells in statevector that overlap with polygon of region boundaries
            w_mask[shape.record[1]] = grid_shape_overlap(statevector, x, y, shape.record[1]) 

    for r in w_mask.columns:
        emis = emissions['EmisCH4_Total'].squeeze() * emissions['AREA']
        emis = clusters_2d_to_1d(statevector, emis)
        w_mask[r] *= emis

    return w_mask


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
    '''Analyze posterior OH statistics'''
    # currently no option for KF

    inversion_result = xr.load_dataset(f'{data_dir}/inversion/inversion_result.nc')
    # Keep only OH elements for analysis
    xhat_OH = inversion_result["xhat"][-2:].to_numpy()
    S_post_OH = inversion_result["S_post"].isel(nvar1=slice(-2,None), nvar2=slice(-2,None)).to_numpy()
    A_OH = inversion_result["A"].isel(nvar1=slice(-2,None), nvar2=slice(-2,None)).to_numpy()

    return xhat_OH, S_post_OH, A_OH


def plot_correlation(w_matrix, kf=False):
    '''Plot error correlation matrix given W'''

    w = w_matrix.T
    inversion_result = xr.load_dataset(f'{data_dir}/inversion/inversion_result.nc')
    # Keep only emission elements for analysis
    xhat_emissions = inversion_result["xhat"][:-2]
    S_post = inversion_result["S_post"].isel(nvar1=slice(None,-2), nvar2=slice(None,-2))
    A = inversion_result["A"].isel(nvar1=slice(None,-2), nvar2=slice(None,-2))

    _, _, pearson, _ = source_attribution(w, xhat_emissions, S_post, A)
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

    attribution = 'sector' if 'sector' in w_matrix else 'region'
    if kf:
        plt.savefig(f'error_corr_{year}{i+1:02d}_{attribution}.png')
    else:
        plt.savefig(f'error_corr_{year}_{attribution}.png')

if __name__ == "__main__":

    year = 2019
    kalman_mode = False
    start_date = f"{year}0101"
    end_date = f"{int(year)+1}0101"
    shapefile_path = "shapefiles/merged.shp" # the merged shapefile is created using make_shapefiles.py

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

            print(f"Monthly total for {year}{i+1:02d}: {w.to_numpy().sum()*86400*31*1e-9} Tg")
            w.to_csv(f'{data_dir}/w_{year}{i+1:02d}.csv', index=False)
            plot_correlation(w, kalman_mode)

    else:
        ds = get_mean_emissions(start_date, end_date, prior_cache_path)
        ds.to_netcdf(f"{data_dir}/HEMCO_diagnostics.{year}.nc")
        # Also check annual mean emissions
        emissions_in_kg_per_s = ds["EmisCH4_Total"] * ds["AREA"]
        total = emissions_in_kg_per_s.sum() * 86400 * 365 * 1e-9
        print(f"Total prior (incl. soil sink): {total.values:.2f} Tg/yr")

        # Save annual mean W sectoral matrix
        w_sectoral = sectoral_matrix(sv, ds)
        w_total = w_sectoral.to_numpy().sum() * 86400 * 365 * 1e-9
        print(f"Total from sectoral W matrix: {w_total:.2f} Tg/yr") # this should be slightly lower than the total prior from above?
        # w.to_csv(f'{data_dir}/w_{year}_annual_sectors.csv', index=False)

        # Save annual mean W regional matrix
        regions = shapefile.Reader(shapefile_path, encoding='windows-1252')
        unique_regions = np.unique([r.record[1] for r in regions.shapeRecords()])
        w_regions_mask = pd.DataFrame(columns=unique_regions)
        w_regional = regional_matrix(sv, ds, regions, w_regions_mask)
        # w_regional_emis.to_csv(f'{data_dir}/w_{year}_annual_regions.csv', index=False)
        w_total = w_regional.to_numpy().sum() * 86400 * 365 * 1e-9
        print(f"Total from regional W matrix: {w_total:.2f} Tg/yr")

        # Plot
        plot_correlation(w_sectoral, kalman_mode)
        plot_correlation(w_regional, kalman_mode)

        # Print OH statistics
        xhat_OH, S_post_OH, A_OH = analyze_OH(data_dir)
        print(f"xhat[OH]: {xhat_OH}")
        print(f"S_post[OH]: {S_post_OH}")
        print(f"Trace of A[OH]: {np.trace(A_OH)}")