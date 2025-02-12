import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
from src.inversion_scripts.utils import plot_field

invdir = f"/n/holylfs06/LABS/jacob_lab2/Lab/mhe"
years = [2019, 2020, 2021, 2022, 2023]
sector = "Wetlands"
oil_gas = True if sector == "OG" else False
waste = True if sector == "Wastewater_Landfills_OtherAnth" else False # combine due to low ability of inversion to separate these sectors

# Load state vector
state_vector = xr.open_dataset(f"{invdir}/Global_2020_annual/StateVector.nc")
state_vector_labels = state_vector["StateVector"]
last_ROI_element = int(
    np.nanmax(state_vector_labels.values) - 0
)
mask = state_vector_labels <= last_ROI_element

emissions_list = []
areas = []

for year in years:
    if year == 2019:
        posterior_ds = xr.load_dataset(f"{invdir}/Global_{year}_burnin/inversion/posterior_ds.nc")
    else:
        posterior_ds = xr.load_dataset(f"{invdir}/Global_{year}_annual/inversion/posterior_ds.nc")

    area = posterior_ds["AREA"]

    if oil_gas:
        posterior = posterior_ds["EmisCH4_Oil"] + posterior_ds["EmisCH4_Gas"]
    elif waste:
        posterior = posterior_ds["EmisCH4_Wastewater"] + posterior_ds["EmisCH4_Landfills"] + posterior_ds["EmisCH4_OtherAnth"]
    else:
        posterior = posterior_ds[f"EmisCH4_{sector}"]

    posterior *= area # convert to kg/s
    emissions_list.append(posterior)

# stack emissions along time dimension and assign years
emissions = xr.concat(emissions_list, dim="year")
emissions = emissions.assign_coords(year=("year", np.array(years)))

# Conversion for Tg/yr2
conversion = (86400 * 365 * 1e-9)

# Fit a linear trend using polyfit
trend = (emissions.where(mask) * conversion).polyfit(dim="year", deg=1)["polyfit_coefficients"].sel(degree=1)

# Plot the linear trend across years
fig = plt.figure(figsize=(12, 8))
plt.rcParams.update({"font.size": 16})
ax = fig.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})

plot_save_path = "sectoral_trend_plots"

plot_field(
    ax,
    trend,
    cmap='RdBu_r',
    lon_bounds=[-170, 167.5],
    lat_bounds=[-60, 80],
    vmin=-0.2,
    vmax=0.2,
    title=f"{sector} trend {years[0]}-{years[-1]}",
    cbar_label=r"$\Delta$ Emissions ($Tg\ a^{-2}$)",
    only_ROI=True,
    state_vector_labels=state_vector_labels,
    last_ROI_element=last_ROI_element,
    is_regional=True,
    save_path=plot_save_path
)