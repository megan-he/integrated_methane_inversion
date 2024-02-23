import numpy as np
import xarray as xr
import datetime
import calendar
from joblib import Parallel, delayed
from src.inversion_scripts.utils import zero_pad_num_hour


def zero_pad_num(n):
    nstr = str(n)
    if len(nstr) == 1:
        nstr = "000" + nstr
    if len(nstr) == 2:
        nstr = "00" + nstr
    if len(nstr) == 3:
        nstr = "0" + nstr
    return nstr

def test_GC_output_for_BC_perturbations(e, nelements, sensitivities):

    """
    Ensures that CH4 boundary condition perturbation in GEOS-Chem is working as intended
    sensitivities = (pert-base)/perturbationBC which should equal 1e-9 inside the perturbed borders
    example: the north boundary is perturbed by 10 ppb
             pert-base=10e-9 mol/mol in the 3 grid cells that have been perturbed
             perturbationBC=10 ppb
             sensitivities = (pert-base)/perturbationBC = 1e-9
    """

    if e == (nelements - 4): # North boundary
        check = np.mean(sensitivities[:,-3:,3:-3])
    elif e == (nelements - 3): # South boundary
        check = np.mean(sensitivities[:,0:3,3:-3])
    elif e == (nelements - 2): # East boundary
        check = np.mean(sensitivities[:,:,-3:])
    elif e == (nelements - 1): # West boundary
        check = np.mean(sensitivities[:,:,0:3])
    assert abs(check - 1e-9) < 1e-11, f"GC CH4 perturb not working... perturbation is off by {abs(check - 1e-9)} mol/mol/ppb"

def calc_sensi(
        nelements, perturbation, startday, endday, run_dirs_pth, run_name, sensi_save_pth, perturbationBC, perturbationOH
):
    """
    Loops over output data from GEOS-Chem perturbation simulations to compute sensitivities
    for the Jacobian matrix.

    Arguments
        nelements      [int]   : Number of state vector elements
        perturbation   [float] : Size of emissions perturbation (e.g., 1.5)
        startday       [str]   : First day of inversion period; formatted YYYYMMDD
        endday         [str]   : Last day of inversion period; formatted YYYYMMDD
        run_dirs_pth   [str]   : Path to directory containing GC Jacobian run directories
        run_name       [str]   : Simulation run name; e.g. 'CH4_Jacobian'
        sensi_save_pth [str]   : Path to save the sensitivity data
        perturbationBC [float] : Size of BC perturbation in ppb (eg. 10.0)
        perturbationOH [float] : Size of OH perturbation in ppb (eg. 1.5)

    Resulting 'sensi' files look like:

        <xarray.Dataset>
        Dimensions:  (grid: 1207, lat: 105, lev: 47, lon: 87)
        Coordinates:
        * lon      (lon) float64 -107.8 -107.5 -107.2 -106.9 ... -81.56 -81.25 -80.94
        * lat      (lat) float64 10.0 10.25 10.5 10.75 11.0 ... 35.25 35.5 35.75 36.0
        * lev      (lev) int32 1 2 3 4 5 6 7 8 9 10 ... 38 39 40 41 42 43 44 45 46 47
        * grid     (grid) int32 1 2 3 4 5 6 7 8 ... 1201 1202 1203 1204 1205 1206 1207
        Data variables:
            sensi    (grid, lev, lat, lon) float32 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0

    Pseudocode summary:

        for each day:
            load the base run SpeciesConc file
            nlon = count the number of longitudes
            nlat = count the number of latitudes
            nlev = count the number of vertical levels
            for each hour:
                base = extract the base run data for the hour
                sensi = np.empty((nelements, nlev, nlat, nlon))
                sensi.fill(np.nan)
                for each state vector element:
                    load the SpeciesConc .nc file for the element and day
                    pert = extract the data for the hour
                    sens = pert - base
                    sensi[element,:,:,:] = sens
                save sensi as netcdf with appropriate coordinate variables
    """
    run_dirs_path = "/n/holylfs05/LABS/jacob_lab/Users/jeast/proj/globalinv/prod/output"
    startday = "20180601"
    endday = "20190201"
    nelements = 3753
    
    # subtract by 1 because here we assume .5 is a +50% perturbation
    perturbation = perturbation - 1

    # Make date range (of months)
    months = []
    dt = datetime.datetime.strptime(startday, "%Y%m%d")
    dt_max = datetime.datetime.strptime(endday, "%Y%m%d")
    dt = dt.replace(day=1)
    while dt < dt_max:
        dt_str = dt.strftime("%Y%m%d")
        months.append(dt_str)
        # Move to the first day of the next month
        _, days_in_month = calendar.monthrange(dt.year, dt.month)
        dt = (dt + datetime.timedelta(days=days_in_month)).replace(day=1)

    # Loop over model data to get sensitivities
    # hours = range(24)
    elements = range(1, nelements) # 1 to 3753

    # For each month
    for m in months:
        # define perturbation months for the current month
        for i, m in enumerate(months):
            if i + 2 < len(months):
                pert_months = months[i:i+3] # looks like ["20180601", "20180701", "20180801"] for june 2018
        for pert in pert_months:
            # Load the base run XCH4 file for each perturbation month
            base_data = xr.load_dataset(
                f"{run_dirs_path}/imi_{m}/inversion/data_converted_nc/out_imi_{m}_{pert}_000000.nc"
            )

            # Count nlat, nlon, nlev
            nlon = len(base_data["lon"])  # 144
            nlat = len(base_data["lat"])  # 91
            base_var = base_data["geoschem_methane"] # Read base data before loop

            # Save this data into numpy array so we don't need to read files in loop
            pert_datas = []
            # For each file (state vector element)
            for e in elements:
                # State vector elements are numbered 1..nelements
                elem = zero_pad_num(e)
                # Load the month 1 XCH4 perturbation file for the current element
                pert_data = xr.open_dataset(
                    f"{run_dirs_path}/imi_{m}/inversion/data_converted_nc/out_imi_{m}_{pert}_00{elem}.nc",
                    chunks='auto'
                )
                pert_datas.append(pert_data)
                pert_data.close()

        # For each hour
        def process(h):
            # Get the base run data for the hour
            base = base_data["geoschem_methane"][h, :, :, :]
            # Initialize sensitivities array
            sensi = np.empty((nelements, nlat, nlon))
            sensi.fill(np.nan)
            # For each state vector element
            for e in elements:
                # Get the data for the current hour
                pert = pert_data["geoschem_methane"][h, :, :, :]
                # Compute and store the sensitivities
                if ((perturbationOH > 0.0) and (e >= nelements-1)):
                    sensitivities = (pert.values - base.values) / perturbationOH
                elif (perturbationBC > 0.0):
                    if ((perturbationOH > 0.0) and (e >= (nelements-5))) or ((perturbationOH <= 0.0) and (e >= (nelements-4))):
                        sensitivities = (pert.values - base.values) / perturbationBC
                        if h != 0: # because we take the first hour on the first day from spinup
                            test_GC_output_for_BC_perturbations(e, nelements, sensitivities)
                else:
                    sensitivities = (pert.values - base.values) / perturbation
                sensi[e, :, :, :] = sensitivities
            # Save sensi as netcdf with appropriate coordinate variables
            sensi = xr.DataArray(
                sensi,
                coords=(
                    np.arange(1, nelements + 1),
                    np.arange(1, nlev + 1),
                    base.lat,
                    base.lon,
                ),
                dims=["element", "lev", "lat", "lon"],
                name="Sensitivities",
            )
            sensi = sensi.to_dataset()
            sensi.to_netcdf(
                f"{sensi_save_pth}/sensi_{d}_{zero_pad_num_hour(h)}.nc",
                encoding={v: {"zlib": True, "complevel": 9} for v in sensi.data_vars},
            )

        results = Parallel(n_jobs=-1)(delayed(process)(hour) for hour in hours)
    print(f"Saved GEOS-Chem sensitivity files to {sensi_save_pth}")


if __name__ == "__main__":
    import sys

    nelements = int(sys.argv[1])
    perturbation = float(sys.argv[2])
    startday = sys.argv[3]
    endday = sys.argv[4]
    run_dirs_pth = sys.argv[5]
    run_name = sys.argv[6]
    sensi_save_pth = sys.argv[7]
    perturbationBC = float(sys.argv[8])
    perturbationOH = float(sys.argv[9])

    calc_sensi(
        nelements,
        perturbation,
        startday,
        endday,
        run_dirs_pth,
        run_name,
        sensi_save_pth,
        perturbationBC,
        perturbationOH
    )
