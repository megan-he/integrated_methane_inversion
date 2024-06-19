import xarray as xr
import yaml
import glob
import pandas as pd
import re
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from sklearn.mixture import GaussianMixture

# with open("config.yml", "r") as f:
#     config = yaml.safe_load(f)

"""
super_observations_and_errors.py
Make the super observations and determine their associated error
"""

inversionMonth = sys.argv[1]
input_dir = "/n/holylfs05/LABS/jacob_lab/Users/mhe"
gc_path = "/n/holylfs05/LABS/jacob_lab/Users/nbalasus/gc_global_sensitivities/gc_global/OutputDir"
# os.makedirs(f"inversion_{inversionMonth}")

if __name__ == "__main__":

    # Get relevant satellite observations
    input_files = sorted(glob.glob(input_dir + f"/2019satellite_observations/input_{inversionMonth}*.nc"))
    times = []
    for file in input_files:
        times.append(re.search(r'input_(.*).nc',file).groups(0)[0])

    # Make a individual observations dataframe
    for idx,time in enumerate(times):
        with xr.open_dataset(input_dir + f"/2019satellite_observations/input_{time}.nc") as input_file,\
            xr.open_dataset(gc_path + f"/output_{time}.nc") as output_file:

            tmp_df = pd.DataFrame({"tropomi_gc_lat_grid_cell": input_file["gc_lat_grid_cell"].values,
                                   "tropomi_gc_lon_grid_cell": input_file["gc_lon_grid_cell"].values,
                                   "tropomi_xch4": input_file["xch4"].values,
                                   "gc_xch4": 1e9*output_file["gc_xch4"].values,
                                   "time": time})
            if idx == 0:
                individual_observations = tmp_df
            else:
                individual_observations = pd.concat([individual_observations,tmp_df], ignore_index=True)

    # Average invididual observations by grid cell
    super_observations = individual_observations.groupby(["tropomi_gc_lat_grid_cell","tropomi_gc_lon_grid_cell","time"]).mean().reset_index()
    super_observations_count = individual_observations.groupby(["tropomi_gc_lat_grid_cell","tropomi_gc_lon_grid_cell","time"]).count().reset_index()
    assert np.array_equal(np.array(super_observations["tropomi_gc_lat_grid_cell"]), np.array(super_observations_count["tropomi_gc_lat_grid_cell"]))
    assert np.array_equal(np.array(super_observations["tropomi_gc_lon_grid_cell"]), np.array(super_observations_count["tropomi_gc_lon_grid_cell"]))
    super_observations["count"] = super_observations_count["tropomi_xch4"]

    # Average individual observations by grid cell and time
    mean_super_observations = individual_observations.groupby(["tropomi_gc_lat_grid_cell","tropomi_gc_lon_grid_cell"]).mean(numeric_only=True).reset_index()
    mean_super_observations_count = individual_observations.groupby(["tropomi_gc_lat_grid_cell","tropomi_gc_lon_grid_cell"]).count().reset_index()
    assert np.array_equal(np.array(mean_super_observations["tropomi_gc_lat_grid_cell"]), np.array(mean_super_observations_count["tropomi_gc_lat_grid_cell"]))
    assert np.array_equal(np.array(mean_super_observations["tropomi_gc_lon_grid_cell"]), np.array(mean_super_observations_count["tropomi_gc_lon_grid_cell"]))
    mean_super_observations["count"] = mean_super_observations_count["tropomi_xch4"]

    # If there are less than 30 individual observations in a grid cell across the entire inversion period, drop this grid cell
    # We won't be able to calculate a meaningful error on this anyway
    mean_super_observations = mean_super_observations.loc[mean_super_observations["count"] >= 30].reset_index(drop=True)

    # Calculate s_k via the residual error method
    # This uses individual observations
    # The result will be one s_k per grid cell
    for idx in mean_super_observations.index:

        # Mean y-F(x) across observations and time per grid cell
        mean_y_minus_Fx = mean_super_observations.loc[idx,"tropomi_xch4"] - mean_super_observations.loc[idx,"gc_xch4"]
        
        # For individual observations
        subset_idx = individual_observations.loc[(individual_observations["tropomi_gc_lat_grid_cell"] == mean_super_observations.loc[idx,"tropomi_gc_lat_grid_cell"]) &\
                                                (individual_observations["tropomi_gc_lon_grid_cell"] == mean_super_observations.loc[idx,"tropomi_gc_lon_grid_cell"])].index
        mean_super_observations.loc[idx,"s_k_from_individual_observations"] = ((individual_observations.loc[subset_idx,"tropomi_xch4"] - individual_observations.loc[subset_idx,"gc_xch4"]) - mean_y_minus_Fx).std()**2

    # Now calculate error variance with the residual method using super observations
    # Instead of variance over a grid cell, calculate variance over a given value of P
    # That is, the number of retrievals averaged into the super-observation
    for idx in mean_super_observations.index:
        
        # Mean y-F(x) across observations and time per grid cell
        mean_y_minus_Fx = mean_super_observations.loc[idx,"tropomi_xch4"] - mean_super_observations.loc[idx,"gc_xch4"]

        # For super observations, calculate the error (epsilon)
        subset_idx = super_observations.loc[(super_observations["tropomi_gc_lat_grid_cell"] == mean_super_observations.loc[idx,"tropomi_gc_lat_grid_cell"]) &\
                                            (super_observations["tropomi_gc_lon_grid_cell"] == mean_super_observations.loc[idx,"tropomi_gc_lon_grid_cell"])].index
        super_observations.loc[subset_idx,"epsilon"] = ((super_observations.loc[subset_idx,"tropomi_xch4"] - super_observations.loc[subset_idx,"gc_xch4"]) - mean_y_minus_Fx)

    # Drop NaNs from super observations (these exists because we dropped individual observations inside a grid cell with less than 30 across inversion period)
    super_observations = super_observations.dropna().reset_index(drop=True)

    # Now make a dataframe where P is the number of retrievals used to make a given super observation
    # Columns are (P), (variance of errors across super observations with this many retrievals), and (number of super observations used in the variance calculation)
    df = pd.DataFrame(columns=["P","error_variance","number_of_super_observations_used_to_calculate_error_variance"])
    P_values = np.sort(super_observations["count"].unique()) # from 1 to 1153
    for idx,P in enumerate(P_values):
        df.loc[idx,"P"] = P
        subset_idx = super_observations.loc[super_observations["count"] == P].index
        df.loc[idx,"error_variance"] = super_observations.loc[subset_idx,"epsilon"].std()**2
        df.loc[idx,"number_of_super_observations_used_to_calculate_error_variance"] = len(super_observations.loc[subset_idx])

    # Drop values where we don't have a lot of super observations
    df_drop_low_density = df.loc[df["number_of_super_observations_used_to_calculate_error_variance"] > 1].reset_index(drop=True)
    
    # Plot number of superobs that are used to get each P
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(df_drop_low_density["P"], df_drop_low_density["number_of_super_observations_used_to_calculate_error_variance"])
    ax.set_yscale('log')
    ax.set_xlabel("P")
    ax.set_ylabel("Number of superobservations used to calculate error variance")
    fig.savefig("test.png")

    # Function for error variance
    def func(P, r_retrieval, sigma_retrieval, sigma_transport):
        return sigma_transport**2 + sigma_retrieval**2*((1-r_retrieval)/P + r_retrieval)

    # Optimiziation
    p0 = [0.2, 20.0, 5.0]
    bounds = ([0.,0.,0.], [1.0, np.inf, np.inf])
    popt, pcov = curve_fit(func, np.array(df_drop_low_density["P"]), np.array(df_drop_low_density["error_variance"]), bounds=bounds, p0=p0)
    print(f"r_retrieval     = {popt[0]:.2f}", flush=True)
    print(f"sigma_retrieval = {popt[1]:.2f}", flush=True)
    print(f"sigma_transport = {popt[2]:.2f}", flush=True)

    # Save a plot to check
    fig,ax = plt.subplots()
    clt = lambda P: 1/P * df.iloc[0]["error_variance"]
    ax.plot(df_drop_low_density["P"], clt(df_drop_low_density["P"]), linestyle="--", color="C1", zorder=0, label="Central Limit Theorem")
    ax.scatter(df_drop_low_density["P"], df_drop_low_density["error_variance"], label="Observational Error Variance", edgecolor="k")
    ax.plot(df_drop_low_density["P"], func(df_drop_low_density["P"], *popt), label="Fit", zorder=0)
    ax.set_xlabel("Number P of retrievals averaged in super-observation")
    ax.set_ylabel("Error variance [ppb$^2$]")
    ax.legend()
    ax.grid(linewidth=0.1)
    fig.savefig(f"inversion_{inversionMonth}/observational_error_{inversionMonth}.png", dpi=300, bbox_inches="tight")

    # Define g(P) to scale sk
    def gP(P):
        return func(P,*popt)/func(1,*popt)

    # Scale sk
    super_observations["gPsk"] = np.nan
    for idx in mean_super_observations.index:
        subset_idx = super_observations.loc[(super_observations["tropomi_gc_lat_grid_cell"] == mean_super_observations.loc[idx,"tropomi_gc_lat_grid_cell"]) &\
                                            (super_observations["tropomi_gc_lon_grid_cell"] == mean_super_observations.loc[idx,"tropomi_gc_lon_grid_cell"])].index
        super_observations.loc[subset_idx,"gPsk"] = mean_super_observations.loc[idx,"s_k_from_individual_observations"] * gP(super_observations.loc[subset_idx,"count"])

    # Save out super observations and their density
    super_observations[["tropomi_gc_lat_grid_cell","tropomi_gc_lon_grid_cell","time","tropomi_xch4","count","gPsk"]].to_csv(f"inversion_{inversionMonth}/super_observations.csv")
    mean_super_observations[["tropomi_gc_lat_grid_cell","tropomi_gc_lon_grid_cell","count"]].to_csv(f"inversion_{inversionMonth}/number_of_super_observations_per_grid_cell.csv")