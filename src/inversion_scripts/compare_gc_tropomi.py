import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import glob
import re
from utils import load_obj, plot_field

def list_files(directory):
    files = glob.glob(f"{directory}/*.pkl")
    files.sort()
    dt = []
    for f in files:
        match = re.search(r'CH4____(.*?)_', f)
        date_time = match.group(1)
        date_time = f"{date_time[:4]}-{date_time[4:6]}-{date_time[6:8]}T{date_time[9:11]}:{date_time[11:13]}:{date_time[13:]}"
        dt.append(np.datetime64(date_time))
    return dt

def match_observations(prior_directory, posterior_directory):

    # Load TROPOMI/GEOS-Chem and Jacobian matrix data from the .nc file
    dat = xr.open_dataset(prior_directory)
    dat.close()

    # Extract the timesteps in a 1d array
    timesteps = dat["geoschem_methane"].coords['time'].values

    # Match to closest posterior timestep (TROPOMI retrieval time)
    posterior_timesteps = list_files(posterior_directory)
    diffs = np.abs(timesteps[:, np.newaxis] - np.array(posterior_timesteps))
    timesteps_match = timesteps[np.argmin(diffs, axis=0)]
    
    time = np.array([])
    lat = np.array([])
    lon = np.array([])
    tropomi_gc_grid = np.array([])
    tropomi_sat_grid = np.array([])
    geos_prior = np.array([])
    geos_posterior = np.array([])
    posterior_lat = np.array([])
    posterior_lon = np.array([])
    observation_count = np.array([])
    observation_count_posterior = np.array([])
    
    # Load GC/TROPOMI data, obs counts, lats, lons. TROPOMI data is mapped onto GC grid already
    for t in range(len(timesteps_match)):

        GC_all = dat["geoschem_methane"].values[t,:,:]
        non_nans = np.where(~np.isnan(GC_all))
        GC = GC_all[non_nans]
        
        tropomi_all = dat["tropomi_methane"].values[t,:,:]
        tropomi_methane_gc_grid = tropomi_all[non_nans]

        tropomi_lat = dat["lat_sat"].values[t,:,:]
        tropomi_lat = tropomi_lat[non_nans]
        tropomi_lon = dat["lon_sat"].values[t,:,:]
        tropomi_lon = tropomi_lon[non_nans]
        
        obs_count_all = dat["observation_count"].values[t,:,:]
        obs_count = obs_count_all[non_nans]

        t_array = np.full_like(tropomi_methane_gc_grid, t)

        # Record lat, lon, tropomi ch4, and geos ch4
        time = np.concatenate((time, t_array))
        lat = np.concatenate((lat, tropomi_lat))
        lon = np.concatenate((lon, tropomi_lon))
        tropomi_gc_grid = np.concatenate((tropomi_gc_grid, tropomi_methane_gc_grid))
        geos_prior = np.concatenate((geos_prior, GC))
        observation_count = np.concatenate((observation_count, obs_count))
    
    # Load posterior GC methane
    post_files = np.sort(os.listdir(posterior_directory))
    for f in post_files:
        pth_posterior = os.path.join(posterior_directory, f)
        obj_posterior = load_obj(pth_posterior)
        obs_GC_posterior = obj_posterior["obs_GC"][:,1]
        tropomi_methane_sat_grid = obj_posterior["obs_GC"][:,0]
        lat_sat = obj_posterior["obs_GC"][:,3]
        lon_sat = obj_posterior["obs_GC"][:,2]
        obs_count_post = obj_posterior["obs_GC"][:,4]
        tropomi_sat_grid = np.concatenate((tropomi_sat_grid, tropomi_methane_sat_grid))
        geos_posterior = np.concatenate((geos_posterior, obs_GC_posterior))
        posterior_lat = np.concatenate((posterior_lat, lat_sat))
        posterior_lon = np.concatenate((posterior_lon, lon_sat))
        observation_count_posterior = np.concatenate((observation_count_posterior, obs_count_post))

    df = pd.DataFrame()
    df["time"] = time
    df["lat"] = lat
    df["lon"] = lon
    df["tropomi"] = tropomi_gc_grid
    df["geos_prior"] = geos_prior
    df["diff_tropomi_prior"] = geos_prior - tropomi_gc_grid
    df["observation_count"] = observation_count

    df_post = pd.DataFrame()
    df_post["geos_posterior"] = geos_posterior
    df_post["diff_tropomi_posterior"] = geos_posterior - tropomi_sat_grid
    df_post["lat"] = posterior_lat
    df_post["lon"] = posterior_lon
    df_post["observation_count"] = observation_count_posterior

    return df, df_post

def plot_comparison(lat_bounds, lon_bounds):

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
        ds_month["tropomi"],
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
        ds_month["diff_tropomi_prior"],
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
        ds_post_rounded["diff_tropomi_posterior"],
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
    import sys

    # run with srun --mem-per-cpu=5G python compare_gc_tropomi.py "YYYYMMDD"
    # make sure sensi files are in data_converted

    month = sys.argv[1] # "YYYYMMDD" string format

    data_dir_post = "./data_visualization_posterior"
    data_dir_prior = f"./data_converted/sensi_{month}_{month}.nc"

    df_month, df_post_month = match_observations(data_dir_prior, data_dir_post)

    visualization_df = df_month.groupby(["lat", "lon"]).mean()
    ds_month = visualization_df.to_xarray()   

    # Round posterior to 2° x 2.5°
    d = {var: df_post_month[var] for var in df_post_month.columns}
    d["lat"] = np.round(df_post_month["lat"]/2.0)*2.0
    d["lon"] = np.round(df_post_month["lon"]/2.5)*2.5
    df_post_rounded = pd.DataFrame(d)
    ds_post_rounded = df_post_rounded.groupby(["lat","lon"]).mean().to_xarray()

    print(
    "Bias in prior     :",
    np.round(
        np.average(
            df_month["diff_tropomi_prior"], weights=df_month["observation_count"]
        ),
        2,
    ),
    "ppb",
    )

    print(
        "Bias in posterior :",
        np.round(
            np.average(
                df_post_rounded["diff_tropomi_posterior"],
                # weights=df_month["observation_count"],
            ),
            2,
        ),
        "ppb",
    )

    # Set up map bounds for plotting
    lat_bounds = [-56.0, 76.0]
    lon_bounds = [-170.0, 167.5]

    # Plot and save figures
    plot_comparison(lat_bounds, lon_bounds)
    print("Done plotting")
    