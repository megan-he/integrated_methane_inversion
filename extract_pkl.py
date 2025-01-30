import numpy as np
import xarray as xr
import glob
import os
import pandas as pd
import re
import sys
from src.inversion_scripts.utils import load_obj


'''
Extract IMI output (.pkl) across full inversion year.
Data is saved as npz
'''

mode = sys.argv[1]

if mode == "data_viz":
    satdat_dir = "/n/holylfs06/LABS/jacob_lab2/Lab/mhe/Global_2020_annual/inversion/data_visualization/"
    save_path = "/n/holylfs05/LABS/jacob_lab/Users/mhe/Global_2020_annual/data_viz_extracted_fullyear/"
elif mode == "data_converted":
    satdat_dir = "/n/holylfs06/LABS/jacob_lab2/Lab/mhe/Global_2020_annual/inversion/data_converted/"
    save_path = "/n/holylfs05/LABS/jacob_lab/Users/mhe/Global_2020_annual/data_converted_extracted_fullyear/"
else:
    print(f"Error. Specify 'data_viz' or 'data_converted'.")
    sys.exit(1)

# Get observed and GEOS-Chem-simulated TROPOMI columns
files = [f for f in np.sort(os.listdir(satdat_dir)) if "TROPOMI" in f]
print(f"Number of pkl files: {len(files)}")
lat = np.array([])
lon = np.array([])
tropomi = np.array([])
geos_prior = np.array([])
obs_count = np.array([])
# iSat = np.array([])
# jSat = np.array([])
time = []


for i, f in enumerate(files):
    # grab start and end date/time from filename
    starttime = re.search(r"(\d{8})", f).group(1) # grab the start time (first consecutive 8 digits)
    starttime_dt = pd.to_datetime(starttime, format='%Y%m%d')

    # Get paths
    pth = os.path.join(satdat_dir, f)
    # Get same file from bc folder
    # Load TROPOMI/GEOS-Chem and Jacobian matrix data from the .pkl file
    obj = load_obj(pth)
    # If there aren't any TROPOMI observations on this day, skip
    if obj["obs_GC"].shape[0] == 0:
        continue
    # Otherwise, grab the TROPOMI/GEOS-Chem data
    obs_GC_temp = obj["obs_GC"]

    # concatenate obs data
    tropomi = np.concatenate((tropomi, obs_GC_temp[:, 0]))
    geos_prior = np.concatenate((geos_prior, obs_GC_temp[:, 1]))
    lon = np.concatenate((lon, obs_GC_temp[:, 2]))
    lat = np.concatenate((lat, obs_GC_temp[:, 3]))
    obs_count = np.concatenate((obs_count, obs_GC_temp[:, 4]))
    # iSat = np.concatenate((iSat, obs_GC_temp[:,4]))
    # jSat = np.concatenate((jSat, obs_GC_temp[:,5]))

    # append obs data to get for full year
    if i == 0:
        obs_GC = obs_GC_temp
    else:
        obs_GC = np.append(obs_GC, obs_GC_temp, axis=0)

    # append time data (same time for each individual file)
    time.extend([starttime] * obs_GC_temp.shape[0])

# convert to numpy array
time = np.array(time)

gc_ch4_prior = {"xch40": geos_prior}
obs_tropomi = {"y": tropomi}
lat = {"lat": lat}
lon = {"lon": lon}
obs_count = {"obs_count": obs_count}
# iSat = {"iSat": iSat}
# jSat = {"jSat": jSat}
time = {"time": time}

np.savez(save_path+f"gc_ch4_prior.npz", **gc_ch4_prior)
np.savez(save_path+f"obs_tropomi.npz", **obs_tropomi)
np.savez(save_path+f"lat.npz", **lat)
np.savez(save_path+f"lon.npz", **lon)
np.savez(save_path+f"obs_count.npz", **obs_count)
# np.savez(save_path+f"iSat.npz", **iSat)
# np.savez(save_path+f"jSat.npz", **jSat)
np.savez(save_path+f"time.npz", **time)

print("Saved all observation data")