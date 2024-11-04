import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import colorcet as cc
from src.inversion_scripts.utils import plot_field

def calc_sectoral_trend(sector_name, year_list, oil_gas=False):

    trend_list = []
    areas = []
    
    # Load posterior datasets for each year
    for year in year_list:
        if year == 2019:
            posterior_ds = xr.load_dataset(f"{invdir}/Global_{year}_annual_edgarv7/inversion/posterior_ds.nc")
        else:
            posterior_ds = xr.load_dataset(f"{invdir}/Global_{year}_annual/inversion/posterior_ds.nc")
        
        if oil_gas:
            posterior_sectoral = posterior_ds["EmisCH4_Oil"] + posterior_ds["EmisCH4_Gas"]
        else:
            posterior_sectoral = posterior_ds[f"EmisCH4_{sector_name}"]
        trend_list.append(posterior_sectoral)
        areas.append(posterior_ds["AREA"])

    # Calculate year-to-year differences
    diff_list = [
        (trend_list[i+1] - trend_list[i]) * areas[i] * 86400 * 365 * 1e-9 # convert to Tg/y
        for i in range(len(trend_list) - 1)
    ]

    # Take average of differences
    avg_diffs = sum(diff_list) / (len(year_list)-1) # kg

    return avg_diffs

def plot_range(vals):

    data_range = np.nanmax(vals) - np.nanmin(vals)
    plot_max = np.nanmax(vals) - 0.1*data_range
    plot_min = -plot_max

    return plot_min, plot_max


if __name__ == "__main__":

    invdir = f"/n/holyscratch01/jacob_lab/mhe"
    years = [2019, 2020, 2021]

    sector = "OG"
    if sector == "OG":
        oil_gas = True

    posterior_sector = calc_sectoral_trend(sector, years, oil_gas)

    # Load state vector
    state_vector = xr.load_dataset(f"{invdir}/Global_2019_annual_edgarv7/StateVector.nc")
    state_vector_labels = state_vector["StateVector"]
    last_ROI_element = int(
        np.nanmax(state_vector_labels.values) - 0
    )
    mask = state_vector_labels <= last_ROI_element

    # Plot posterior emissions
    fig = plt.figure(figsize=(8, 8))
    plt.rcParams.update({"font.size": 16})
    ax = fig.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})

    min_trend, max_trend = plot_range(posterior_sector)
    plot_save_path = "sectoral_trend_plots"

    plot_field(
        ax,
        posterior_sector,
        cmap='RdBu_r',
        lon_bounds=[-170, 167.5],
        lat_bounds=[-60, 80],
        vmin=-0.3,
        vmax=0.3,
        title=f"Posterior {sector.lower()} trend {years[0]}-{years[-1]}",
        cbar_label="Emissions trend (Tg/a)",
        only_ROI=True,
        state_vector_labels=state_vector_labels,
        last_ROI_element=last_ROI_element,
        is_regional=False,
        save_path=plot_save_path
    )