import xarray as xr
import pandas as pd
from scipy import constants
import numpy as np
import matplotlib.pyplot as plt
import calendar

'''
Before running this script, need to turn on CH4, BoundaryConditions, and StateMet collections in HISTORY.rc in GEOS-Chem run.
'''

def calculate_methane_lifetime_to_oh(year, dirname):
    '''
    Returns average methane lifetime to tropospheric OH (years).
    '''

    # Open CH4 collection (monthly)
    ds = xr.open_mfdataset(dirname+f"GEOSChem.CH4.{year}*_0000z.nc4")
    sec_in_year = (365 + calendar.isleap(pd.to_datetime(ds["time"].values[0]).year)) * 24 * 60 * 60

    loss_to_OH_in_trop = ds["LossCH4byOHinTrop"].sum(('lev','lat','lon')).values.mean() # take mean over 12 months
    loss_to_OH_in_trop = loss_to_OH_in_trop * sec_in_year * 1e-9 # [Tg/y]

    print(f"Loss to OH in troposphere: {loss_to_OH_in_trop} Tg")

    # Open BoundaryConditions collection (daily)
    # this is the same as SpeciesConc collection but with the first hour (t=0) outputted
    BC_ds = xr.open_mfdataset(dirname+f"GEOSChem.BoundaryConditions.*_0000z.nc4")

    mol_dry_air = BC_ds['Met_AD'] * 1e3/29
    mol_ch4 = BC_ds['SpeciesBC_CH4'] * mol_dry_air
    mol_ch4_sum = mol_ch4.sum(dim=['lat', 'lon', 'lev']).load() # explicitly load data or else these arrays are evaluated lazily with dask
    Tg_ch4 = mol_ch4_sum * 16/1e12

    # Take avg across whole year
    avg_ch4 = Tg_ch4.mean().values
    print(f"Average CH4 mass in atmosphere for {year}: {avg_ch4:.2f} Tg")

    ch4_lifetime_to_oh = avg_ch4 / loss_to_OH_in_trop
    
    return ch4_lifetime_to_oh

def calculate_interhemispheric_ratio(year, dirname):
    '''
    Returns North to South hemispheric ratio for air mass weighted OH.
    '''

    # Open CH4 and StateMet collections
    oh_ds = xr.open_mfdataset(dirname + f'GEOSChem.CH4.{year}*_0000z.nc4')
    met_ds = xr.open_mfdataset(dirname + f'GEOSChem.StateMet.{year}*')

    # Convert Met_AIRDEN from kg/m3 to molec/cm3
    met_ds['Met_AIRDEN'] = (met_ds['Met_AIRDEN'] * (1/28.9647)*1e3*constants.N_A*1e-6).astype(np.float64)
    met_ds['Met_AIRDEN'].attrs['units'] = 'molec cm-3'

    # Calculate molecules of air
    met_ds['airmass'] = (met_ds['Met_AIRDEN'] * met_ds['Met_AIRVOL'] * 1e6).astype(np.float64)
    met_ds['airmass'].attrs['units'] = 'molec'

    # Calculate air mass-weighted OH
    oh_ds['xohmass'] = oh_ds['OHconcAfterChem'] * met_ds['airmass']
    oh_ds['xoh_airmasswgt'] = (oh_ds['xohmass'].sum(('lev','lat','lon')) / met_ds['airmass'].sum(('lev','lat','lon'))) / 1e5

    # Set weights for each hemisphere
    nh_wgt = xr.where(oh_ds['lat'] > 0, 1, xr.where(oh_ds['lat'] == 0, 0.5, 0))
    sh_wgt = xr.where(oh_ds['lat'] < 0, 1, xr.where(oh_ds['lat'] == 0, 0.5, 0))
    nh_wgt = nh_wgt.broadcast_like(oh_ds['xohmass'])
    sh_wgt = sh_wgt.broadcast_like(oh_ds['xohmass'])

    # Calculate air mass-weighted OH for each hemisphere
    NH_oh_airmasswgt = ((
        (oh_ds['xohmass'] * nh_wgt).sum(('lev', 'lat', 'lon')) /
        (met_ds['airmass'] * nh_wgt).sum(('lev', 'lat', 'lon'))
    ) / 1e5).mean()

    SH_oh_airmasswgt = ((
        (oh_ds['xohmass'] * sh_wgt).sum(('lev', 'lat', 'lon')) /
        (met_ds['airmass'] * sh_wgt).sum(('lev', 'lat', 'lon'))
    ) / 1e5).mean()

    NH_SH_ratio = NH_oh_airmasswgt / SH_oh_airmasswgt
    return NH_SH_ratio.values


if __name__ == "__main__":

    # Set directories
    # year = 2019
    # directory = f'/n/holylfs05/LABS/jacob_lab/Users/mhe/Global_{year}_burnin/posterior_run/OutputDir/'
    # directory = f'/n/netscratch/jacob_lab/Lab/mhe/Global_{year}_annual_edgarv7/jacobian_runs/Global_2019_annual_edgarv7_0000/OutputDir/'

    year = 2020
    directory = f'/n/holylfs06/LABS/jacob_lab2/Lab/mhe/Global_{year}_annual/posterior_run/OutputDir/'

    methane_lifetime = calculate_methane_lifetime_to_oh(year, directory)
    print(f"CH4 lifetime to OH: {methane_lifetime} yr")

    ratio = calculate_interhemispheric_ratio(year, directory)
    print(f"N/S ratio: {ratio}")