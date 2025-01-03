import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import colorcet as cc
from src.inversion_scripts.utils import plot_field

def calc_sectoral_trend(sector_name, year_list, oil_gas=False, wastewater_landfills=False):

    trend_list = []
    areas = []
    
    # Load posterior datasets for each year
    for year in year_list:
        if year == 2019:
            posterior_ds = xr.load_dataset(f"/n/holylfs05/LABS/jacob_lab/Users/mhe/Global_{year}_burnin/inversion/posterior_ds.nc")
        else:
            posterior_ds = xr.load_dataset(f"{invdir}/Global_{year}_annual/inversion/posterior_ds.nc")
        
        if oil_gas:
            posterior_sectoral = posterior_ds["EmisCH4_Oil"] + posterior_ds["EmisCH4_Gas"]
        elif wastewater_landfills:
            posterior_sectoral = posterior_ds["EmisCH4_Wastewater"] + posterior_ds["EmisCH4_Landfills"] + posterior_ds["EmisCH4_OtherAnth"]
        else:
            posterior_sectoral = posterior_ds[f"EmisCH4_{sector_name}"]
        trend_list.append(posterior_sectoral)
        areas.append(posterior_ds["AREA"])

    trend_list_Tg_y = [
        trend_list[i] * areas[i] * 86400 * 365 * 1e-9 # convert to Tg/y
        for i in range(len(trend_list) - 1)
    ]

    # Calculate year-to-year differences
    diff_list = [
        (trend_list[i+1] - trend_list[i]) * areas[i] * 86400 * 365 * 1e-9 # convert to Tg/y
        for i in range(len(trend_list) - 1)
    ]

    # Take average of differences
    avg_diffs = sum(diff_list) / (len(year_list)-1) # Tg

    # Set very small values to NaNs. then calculate percent difference between latest year and first year
    last_year = trend_list_Tg_y[-1].where(trend_list_Tg_y[-1] > 0.001, np.nan) # a bit arbitrary threshold to make visualization look better
    first_year = trend_list_Tg_y[0].where(trend_list_Tg_y[0] > 0.001, np.nan)

    diff_percent = (last_year - first_year)/first_year * 100

    return avg_diffs, diff_percent


if __name__ == "__main__":

    invdir = f"/n/holylfs06/LABS/jacob_lab2/Lab/mhe"
    years = [2019, 2020, 2021, 2022, 2023]

    sector = "Wetlands"
    oil_gas = True if sector == "OG" else False
    wastewater_landfills = True if sector == "Wastewater_Landfills_OtherAnth" else False # combine due to low ability of inversion to separate these sectors

    posterior_sector_absolute, posterior_sector_percent = calc_sectoral_trend(sector, years, oil_gas, wastewater_landfills)

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

    plot_save_path = "sectoral_trend_plots"

    plot_field(
        ax1,
        posterior_sector_absolute,
        cmap='RdBu_r',
        lon_bounds=[-170, 167.5],
        lat_bounds=[-60, 80],
        vmin=-0.3,
        vmax=0.3,
        title=f"Absolute {sector.lower()} trend {years[0]}-{years[-1]}",
        cbar_label="Tg/a",
        only_ROI=True,
        state_vector_labels=state_vector_labels,
        last_ROI_element=last_ROI_element,
        is_regional=False,
    )

    plot_field(
        ax2,
        posterior_sector_percent,
        cmap='RdBu_r',
        lon_bounds=[-170, 167.5],
        lat_bounds=[-60, 80],
        vmin=-100,
        vmax=100,
        title=f"Relative {sector.lower()} trend {years[0]}-{years[-1]}",
        cbar_label="%",
        only_ROI=True,
        state_vector_labels=state_vector_labels,
        last_ROI_element=last_ROI_element,
        is_regional=False,
        save_path=plot_save_path
    )