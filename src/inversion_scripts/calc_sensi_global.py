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

def observation_months(current_month):
    obs_months = []
    
    month1 = datetime.datetime.strptime(current_month, "%Y%m%d")

    last_day = calendar.monthrange(month1.year, month1.month)[1]

    # Calculate the next two months
    month2 = month1 + datetime.timedelta(days=last_day)
    month3 = month2 + datetime.timedelta(days=calendar.monthrange(month2.year, month2.month)[1])

    obs_months = [month1.strftime("%Y%m%d"), month2.strftime("%Y%m%d"), month3.strftime("%Y%m%d")]

    return obs_months

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

    # subtract by 1 because here we assume .5 is a +50% perturbation
    perturbation = perturbation - 1

    # Make date range for month with perturbation
    pert_months = []
    dt = datetime.datetime.strptime(startday, "%Y%m%d")
    dt_max = datetime.datetime.strptime(endday, "%Y%m%d")
    dt = dt.replace(day=1)
    while dt < dt_max:
        dt_str = dt.strftime("%Y%m%d")
        pert_months.append(dt_str)
        # Move to the first day of the next month
        _, days_in_month = calendar.monthrange(dt.year, dt.month)
        dt = (dt + datetime.timedelta(days=days_in_month)).replace(day=1)
    print(f"months: {pert_months}")

    # Loop over model data to get sensitivities
    elements = range(nelements) # 0 to 3752

    def process(m):
        print(f"perturbation month: {pert_months}")
        obs_months = observation_months(m)
        print(f"observation months {obs_months} for pert month {m}")

        def process_obs(o):
            print(f"o: {o}")
            # Load the base run XCH4 file for each observation month
            base_data = xr.open_dataset(
                f"{run_dirs_pth}/imi_{m}/inversion/data_converted_nc/out_imi_{m}_{o}_000000.nc"
            )
            
            # Count nlat, nlon, timestep
            nlon = len(base_data["lon"])  # 144
            nlat = len(base_data["lat"])  # 91
            base = base_data["geoschem_methane"] # Read base data before loop
            K = base_data["K"]
            # print("loaded K")
            base_data.close()
            # print("closed base data")

            # Save this data into numpy array so we don't need to read files in loop
            # pert_datas = []
            # For each state vector element
            for e in elements:
                # State vector elements are numbered 1..nelements
                elem = zero_pad_num(e + 1)
                # Load the month 1 XCH4 perturbation file for the current element
                pert_data = xr.open_dataset(
                    f"{run_dirs_pth}/imi_{m}/inversion/data_converted_nc/out_imi_{m}_{o}_00{elem}.nc"
                )
                pert = pert_data["geoschem_methane"]
                pert_data.close()

                # Compute and store the sensitivities
                # if ((perturbationOH > 0.0) and (e >= nelements-1)):
                #     sensitivities = (pert.values - base.values) / perturbationOH
                # elif (perturbationBC > 0.0):
                #     if ((perturbationOH > 0.0) and (e >= (nelements-5))) or ((perturbationOH <= 0.0) and (e >= (nelements-4))):
                #         sensitivities = (pert.values - base.values) / perturbationBC # this is np array with time, lat, lon dimensions
                #         # if h != 0: # because we take the first hour on the first day from spinup
                #         #     test_GC_output_for_BC_perturbations(e, nelements, sensitivities)
                if (perturbation > 0.0):
                    sensitivities = (pert.values - base.values) / perturbation
                K[:, :, :, e] = sensitivities
            
            # Save sensi as netcdf
            encoding_dict = {}
            for var in base_data.data_vars:
                encoding_dict[var] = {"zlib": True, "complevel": 1}
            print(encoding_dict)
            with xr.set_options(keep_attrs=True):
                base_data.to_netcdf(
                    f"{sensi_save_pth}/sensi_{m}_{o}.nc",
                    format='NETCDF4',
                    encoding=encoding_dict,
                )
            print("converted to nc")

        Parallel(n_jobs=-1)(delayed(process_obs)(o) for o in obs_months)

    results = Parallel(n_jobs=-1)(delayed(process)(m) for m in pert_months)
    print(f"Saved GEOS-Chem sensitivity files to {sensi_save_pth}")


if __name__ == "__main__":
    import sys

    # nelements = int(sys.argv[1])
    # perturbation = float(sys.argv[2])
    # startday = sys.argv[3]
    # endday = sys.argv[4]
    # run_dirs_pth = sys.argv[5]
    # run_name = sys.argv[6]
    # sensi_save_pth = sys.argv[7]
    # perturbationBC = float(sys.argv[8])
    # perturbationOH = float(sys.argv[9])

    nelements = 3753
    perturbation = 1.5
    startday = "20191101"
    endday = "20200101"
    run_dirs_pth = "/n/holylfs05/LABS/jacob_lab/Users/jeast/proj/globalinv/prod/output"
    run_name = None
    sensi_save_pth = "/n/holyscratch01/jacob_lab/mhe/calc_sensi_test"
    perturbationBC = None
    perturbationOH = None

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
