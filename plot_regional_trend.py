import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import colorcet as cc
from src.inversion_scripts.utils import plot_field

def calc_regional_trend(year_list):

    trend_list = []
    areas = []
    
    # Load posterior datasets for each year
    for year in year_list:
        if year == 2019:
            posterior_ds = xr.load_dataset(f"/n/holylfs05/LABS/jacob_lab/Users/mhe/Global_{year}_burnin/inversion/posterior_ds.nc")
        else:
            posterior_ds = xr.load_dataset(f"{invdir}/Global_{year}_annual/inversion/posterior_ds.nc")

        posterior = posterior_ds["EmisCH4_Total"]
        trend_list.append(posterior)
        areas.append(posterior_ds["AREA"])

    # Calculate year-to-year differences
    diff_list = [
        (trend_list[i+1] - trend_list[i]) * areas[i] * 86400 * 365 * 1e-9 # convert to Tg/y
        for i in range(len(trend_list) - 1)
    ]

    # Take average of differences
    avg_diffs = sum(diff_list) / (len(year_list)-1) # Tg

    # Calculate year-to-year percent differences
    percent_diff = xr.where(trend_list[0] != 0, ((trend_list[-1] - trend_list[0]) / trend_list[0]) * 100, np.nan)

    return avg_diffs, percent_diff

if __name__ == "__main__":

    invdir = f"/n/holylfs06/LABS/jacob_lab2/Lab/mhe"
    years = [2019, 2020, 2021, 2022, 2023]

    posterior_sector_absolute, posterior_sector_percent = calc_regional_trend(years)

    # Load state vector
    state_vector = xr.load_dataset(f"{invdir}/Global_2020_annual/StateVector.nc")
    state_vector_labels = state_vector["StateVector"]
    last_ROI_element = int(
        np.nanmax(state_vector_labels.values) - 0
    )
    mask = state_vector_labels <= last_ROI_element

    # Plot posterior emissions
    fig = plt.figure(figsize=(24, 8))
    plt.rcParams.update({"font.size": 16})
    ax1, ax2 = fig.subplots(1, 2, subplot_kw={"projection": ccrs.PlateCarree()})

    plot_save_path = "regional_trend_plots"

    plot_field(
        ax1,
        posterior_sector_absolute,
        cmap='RdBu_r',
        lon_bounds=[-170, 167.5],
        lat_bounds=[-60, 80],
        vmin=-0.4,
        vmax=0.4,
        title=f"Absolute emissions trend {years[0]}-{years[-1]}",
        cbar_label="Tg/y",
        only_ROI=True,
        state_vector_labels=state_vector_labels,
        last_ROI_element=last_ROI_element,
        is_regional=True
    )

    plot_field(
        ax2,
        posterior_sector_percent,
        cmap='RdBu_r',
        lon_bounds=[-170, 167.5],
        lat_bounds=[-60, 80],
        vmin=-200,
        vmax=200,
        title=f"Relative emissions trend {years[0]}-{years[-1]}",
        cbar_label="%",
        only_ROI=True,
        state_vector_labels=state_vector_labels,
        last_ROI_element=last_ROI_element,
        is_regional=True,
        save_path=plot_save_path
    )