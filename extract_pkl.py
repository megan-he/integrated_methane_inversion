import numpy as np
import os
import pandas as pd
import re
import sys
from src.inversion_scripts.utils import load_obj

def extract(satdat_dir, save_path, year, is_data_converted):
    '''
    Extracts parameters from inversion/ folders into a npz file (one per parameter).

    Args:
    - satdat_dir: path to input inversion data
    - save_path: path to save extracted data
    - year: year of inversion
    - is_data_converted: are we processing "data_converted"?
    '''

    # Get observed and GEOS-Chem-simulated TROPOMI columns
    files = [f for f in np.sort(os.listdir(satdat_dir)) if "TROPOMI" in f]
    print(f"Processing {'data_converted' if is_data_converted else 'data_visualization'} for {year}")
    print(f"Number of pkl files: {len(files)}")
    
    lat = np.array([])
    lon = np.array([])
    tropomi = np.array([])
    geos_prior = np.array([])
    obs_count = np.array([])
    time = []

    # Exclude problematic date range if 2022 or 2023
    if year == 2022:
        date_range = pd.date_range(start='2022-07-25', end='2022-08-24')
    elif year == 2023:
        date_range = pd.date_range(start='2023-07-25', end='2023-09-01')

    for i, f in enumerate(files):
        # get date from filename
        starttime = re.search(r"(\d{8})", f).group(1)  # Grab the start time
        starttime_dt = pd.to_datetime(starttime, format='%Y%m%d')

        if year in [2022, 2023] and starttime_dt in date_range:
            print(f"Skipping file due to problematic date range: {f}")
            continue

        # Get paths
        pth = os.path.join(satdat_dir, f)

        # Load TROPOMI/GEOS-Chem and Jacobian matrix data from the .pkl file
        obj = load_obj(pth)

        # If no observations, skip
        if obj["obs_GC"].shape[0] == 0:
            continue
        
        obs_GC_temp = obj["obs_GC"]

        if is_data_converted:
            if "20241108T" in pth:
                print(pth)
                print(obs_GC_temp[obs_GC_temp[:, 0] < 1850, :].shape)
                
                # Create a boolean mask where the first column (tropomi y) is greater than 1850
                mask = obs_GC_temp[:, 0] > 1850
                
                # Apply the mask to both arrays so they remain consistent
                obs_GC_temp = obs_GC_temp[mask, :]
                print("removed outlier")

            elif ("20240131" in pth) or ("20240217" in pth):
                print(pth)
                print(obs_GC_temp[obs_GC_temp[:, 0] < 1860, :].shape)
                
                mask = obs_GC_temp[:, 0] > 1860
                obs_GC_temp = obs_GC_temp[mask, :]
                print("removed outlier")

            ind = np.where(
                (obs_GC_temp[:, 2] >= -180) & (obs_GC_temp[:, 2] <= 177.5) &
                (obs_GC_temp[:, 3] >= -60) & (obs_GC_temp[:, 3] <= 88) &
                (np.round(obs_GC_temp[:, 4]) > 0)
            )[0]
        else:
            ind = np.where(
                (obs_GC_temp[:, 2] >= -180) & (obs_GC_temp[:, 2] <= 177.5) &
                (obs_GC_temp[:, 3] >= -60) & (obs_GC_temp[:, 3] <= 88)
            )[0]

        # Skip if no data in bounds
        if len(ind) == 0:
            continue

        # TROPOMI and GEOS-Chem data within bounds
        obs_GC_temp = obs_GC_temp[ind, :]

        # Concatenate extracted data
        tropomi = np.concatenate((tropomi, obs_GC_temp[:, 0]))
        geos_prior = np.concatenate((geos_prior, obs_GC_temp[:, 1]))
        lon = np.concatenate((lon, obs_GC_temp[:, 2]))
        lat = np.concatenate((lat, obs_GC_temp[:, 3]))
        obs_count = np.concatenate((obs_count, obs_GC_temp[:, 4]))

        # Append to full-year dataset
        if i == 0:
            obs_GC = obs_GC_temp
        else:
            obs_GC = np.append(obs_GC, obs_GC_temp, axis=0)

        # Append time data
        time.extend([starttime] * obs_GC_temp.shape[0])

    # Convert to numpy arrays
    time = np.array(time)

    # Save extracted data
    np.savez(save_path + "gc_ch4_posterior.npz", xch40=geos_prior)
    np.savez(save_path + "obs_tropomi.npz", y=tropomi)
    np.savez(save_path + "lat.npz", lat=lat)
    np.savez(save_path + "lon.npz", lon=lon)
    np.savez(save_path + "obs_count.npz", obs_count=obs_count)
    np.savez(save_path + "time.npz", time=time)

    print(f"Saved extracted data to {save_path}")

if __name__ == "__main__":
    year = int(sys.argv[1])

    # Define paths
    if year == 2019:
        base_path = f"/n/holylfs06/LABS/jacob_lab2/Lab/mhe/Global_{year}_annual_ResMefix_2/inversion/"
        save_base = f"/n/holylfs05/LABS/jacob_lab/Users/mhe/Global_{year}_annual_ResMefix_2/"
    else:
        base_path = f"/n/holylfs06/LABS/jacob_lab2/Lab/mhe/Global_{year}_annual/inversion/"
        save_base = f"/n/holylfs05/LABS/jacob_lab/Users/mhe/Global_{year}_annual/"

    data_viz_folder = base_path + "data_visualization/"
    save_viz_folder = save_base + "data_viz_extracted_fullyear/"
    
    data_converted_folder = base_path + "data_converted_posterior/"
    save_converted_folder = save_base + "data_posterior_extracted_fullyear/"

    # extract(data_viz_folder, save_viz_folder, year, is_data_converted=False)
    extract(data_converted_folder, save_converted_folder, year, is_data_converted=True)
