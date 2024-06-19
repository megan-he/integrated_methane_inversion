import numpy as np
import pandas as pd
from netCDF4 import Dataset
import xarray as xr
import multiprocessing
import glob
import re
import os
import sys
from datetime import datetime

storage_dir = "/n/holylfs05/LABS/jacob_lab/Users/mhe/"
os.makedirs(storage_dir+"2019satellite_observations")

def process_list_of_netcdf_files(subset_of_files, month):

    # Loop through all of the files and only keep the observations
    # that are north of 60°S, are not over water, and are not the
    # problematic coastal pixels. Also make sure that we only use
    # observations within the month we are processing.
    df = pd.DataFrame()
    for idx,file in enumerate(subset_of_files):
        with Dataset(file) as ds:

            valid = (ds["latitude"][:] > -60)
            sc = (ds["surface_classification"][:] & 0x03).astype(int)
            valid &= ~((sc == 3) |
                       ((sc == 2) & (ds["chi_square_SWIR"][:] > 20000)))
            valid &= (sc != 1)
            start_date = pd.to_datetime(month, format="%Y%m")
            end_date = start_date + pd.DateOffset(months=1)
            f = "%Y-%m-%dT%H:%M:%S.%fZ"
            valid &= pd.to_datetime(ds["time_utc"][:], format=f) >= start_date
            valid &= pd.to_datetime(ds["time_utc"][:], format=f) < end_date

            tmp_df = pd.DataFrame(
                {"latitude": ds["latitude"][:][valid],
                 "longitude": ds["longitude"][:][valid],
                 "xch4": ds["methane_mixing_ratio_blended"][:][valid],
                 "surface_pressure": ds["surface_pressure"][:][valid],
                 "pressure_interval": ds["pressure_interval"][:][valid],
                 "time_utc": pd.to_datetime(ds["time_utc"][:][valid], format=f),
                 "latitude_bounds": list(ds["latitude_bounds"][:][valid]),
                 "longitude_bounds": list(ds["longitude_bounds"][:][valid]),
                 "averaging_kernel": list(ds["column_averaging_kernel"][:][valid]),
                 "methane_profile_apriori": list(ds["methane_profile_apriori"][:][valid]),
                 "dry_air_subcolumns": list(ds["dry_air_subcolumns"][:][valid])})

            df = pd.concat([df, tmp_df], ignore_index=True)

    return df

if __name__ == "__main__":
        
    # Make a list of months from 2019-01 to 2019-12. These can be changed
    # if you want to expand the dates you run your simulation for.
    months = [str(np.datetime64("2019-01") + np.timedelta64(i, "M")).
              replace("-","") for i in range(12)]

    for month in months:

        # Determine which files could contain data for this month. Then,
        # split those files into parts so that we can use multiple cores
        # each process a chunk of the files.
        satellite_dir = "/n/holylfs05/LABS/jacob_lab/Lab/imi/ch4/blended/"
        files = [f for f in sorted(glob.glob(satellite_dir + "*.nc")) if 
                month in re.search(r'_(\d{6})\d{2}T\d{6}_(\d{6})\d{2}T\d{6}_',
                os.path.basename(f)).groups()]
        split_files = np.array_split(files, multiprocessing.cpu_count())
        inputs = [(subset_of_files, month) for subset_of_files in split_files]

        # Call the function "process_list_of_netcdf_files" to filter the
        # data and get only the variables that we need. Return the data
        # in a pandas DataFrame. Use multiple cores.
        with multiprocessing.Pool() as pool:
            results = pool.starmap(process_list_of_netcdf_files, inputs)
            pool.close()
            pool.join()
        df = pd.concat(results, ignore_index=True)

        # Print summary information about observations in this month
        print(f"Summary for {month}")
        print(f"Observations   --> {len(df)}", flush=True)
        print(f"Min XCH4       --> {df['xch4'].min():.3f}", flush=True)
        print(f"Max XCH4       --> {df['xch4'].max():.3f}", flush=True)
        print(f"Min time       --> {df['time_utc'].min()}", flush=True)
        print(f"Max time       --> {df['time_utc'].max()}", flush=True)
        print(f"Min latitude   --> {df['latitude'].min():.3f}", flush=True)
        print(f"Max latitude   --> {df['latitude'].max():.3f}", flush=True)
        print(f"Min longitude  --> {df['longitude'].min():.3f}", flush=True)
        print(f"Max longitude  --> {df['longitude'].max():.3f}\n", flush=True)

        # Now we need to write a netCDF file for each hour that we 
        # have data. For example, "input_20180501T01.nc". This is
        # because GEOS-Chem will look for a satellite file when
        # ITS_A_NEW_HOUR is true in main.F90. This is like applying
        # the operator in post-processing to hourly CH4 fields.
        df["hour_floor"] = df["time_utc"].dt.floor("h")
        unique_hours = df["hour_floor"].unique()

        # This function will subset our monthly data to only that
        # within the given hour. Additionally, it figures out which
        # 2° x 2.5° grid cell the observation falls inside of.
        def write_netcdf_file_for_unique_hour(unique_hour):

            # The latitude counts by 2.0° except at the poles where
            # it is -89.5° and 89.5°. The longitude counts by 2.5°.
            gc_lat_values = np.arange(-88.0, 90.0, 2.0)
            gc_lat_values = np.insert(gc_lat_values, 0, -89.5)
            gc_lat_values = np.append(gc_lat_values, 89.5)
            gc_lon_values = np.arange(-180.0, 177.5+2.5, 2.5)

            subset = df[df["hour_floor"] == unique_hour].reset_index(drop=True)
            subset["gc_lat_grid_cell"] = np.nan
            subset["gc_lon_grid_cell"] = np.nan

            # GEOS-Chem grid cells are defined by their centers
            # (gc_lat_values and gc_lon_values) and +/- half
            # of the lat and lon resolutions. If the center of
            # the satellite observations is inside of the GEOS-Chem
            # grid cell, it is assigned to it.
            lat_res = 2.0
            lon_res = 2.5
            for lat in gc_lat_values:
                for lon in gc_lon_values:
                    lat_mask = (subset["latitude"] > (lat - lat_res/2)) &\
                               (subset["latitude"] <= (lat + lat_res/2))
                    lon_mask = (subset["longitude"] > (lon - lon_res/2)) &\
                               (subset["longitude"] <= (lon + lon_res/2))
                    # The GEOS-Chem grid ends with a box that extends from
                    # 176.25° to 178.75°. Extend this to 180° to get all of
                    # the satellite observations.
                    if (lon == 177.5):
                        lon_mask = (subset["longitude"] > (lon - lon_res/2)) &\
                                   (subset["longitude"] <= (180))
                    grid_cell_mask = lat_mask & lon_mask
                    assert subset.loc[grid_cell_mask, "gc_lat_grid_cell"].isnull().all()
                    assert subset.loc[grid_cell_mask, "gc_lon_grid_cell"].isnull().all()
                    subset.loc[grid_cell_mask, "gc_lat_grid_cell"] = lat
                    subset.loc[grid_cell_mask, "gc_lon_grid_cell"] = lon

            # Make sure there are no NaNs
            assert not subset.isnull().values.any()

            # Calculate pressure levels
            pressure_levels = np.zeros((len(subset),13))
            pressure_levels[:,0] = subset["surface_pressure"]
            for i in range(12):
                pressure_levels[:,i+1] = pressure_levels[:,i] - subset["pressure_interval"]
            pressure_levels = np.flip(pressure_levels, axis=1)

            # Make Dataset
            ds = xr.Dataset(
                data_vars=dict(
                    latitude=(["nobs"],subset["latitude"]),
                    longitude=(["nobs"],subset["longitude"]),
                    gc_lat_grid_cell=(["nobs"],subset["gc_lat_grid_cell"]),
                    gc_lon_grid_cell=(["nobs"],subset["gc_lon_grid_cell"]),
                    xch4=(["nobs"],subset["xch4"]),
                    pressure_levels=(["nobs","nlev"],pressure_levels),
                    methane_profile_apriori=(["nobs","nlay"],
                                            np.array(subset["methane_profile_apriori"].to_list())),
                    column_averaging_kernel=(["nobs","nlay"],
                                            np.array(subset["averaging_kernel"].to_list())),
                    dry_air_subcolumns=(["nobs","nlay"],
                                        np.array(subset["dry_air_subcolumns"].to_list())),
                    latitude_bounds=(["nobs","ncor"],
                                     np.array(subset["latitude_bounds"].to_list())),
                    longitude_bounds=(["nobs","ncor"],
                                      np.array(subset["longitude_bounds"].to_list()))),
                coords=dict(
                    nobs=(["nobs"], np.arange(0,len(subset))),
                    nlev=(["nlev"], np.arange(0,13)),
                    nlay=(["nlay"], np.arange(0,12)),
                    ncor=(["ncor"], np.arange(0,4)))
                    )

            unique_hour_dt = pd.to_datetime(str(unique_hour)) # For handling the string conversion below

            # Save the netCDF file to your storage directory
            ds.to_netcdf(storage_dir+f"2019satellite_observations/"
                                    f"input_{unique_hour_dt.strftime('%Y%m%dT%H')}.nc")

        # Write the files using multiple cores
        with multiprocessing.Pool() as pool:
            pool.map(write_netcdf_file_for_unique_hour, unique_hours)
            pool.close()
            pool.join()