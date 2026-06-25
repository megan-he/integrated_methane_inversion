#demo: python -u regional_ramping.py -d "/n/holylfs05/LABS/jacob_lab/Users/drewpendergrass/CHEEREIO/METHANE_paper_v02_gammapoint1/postprocess/SNAPSHOT_combined_HEMCO_diagnostics.nc" -type "ensemble" -variable "EmisCH4_Total" -time_factor 31536000 -area_factor 1000000 -mass_factor 1000000000 -med True -y "2019 2020 2021 2022" -fix22 True -global True -o "/n/holylfs05/LABS/jacob_lab/Users/drewpendergrass/CHEEREIO/NRT_methane_paper/2024_07_draft2/FIG8_regional_ramping" -g "2.0x2.5" -l "Canada CONUS Cen.Am. So.Am. Europe Russia China Japan/Koreas SE.Asia Oceania S.Asia C.Asia N.Afr./Mid.East Subsah.Afr." -p "Canada CONUS Cen.Am. So.Am. Europe Russia China Japan/Koreas SE.Asia Oceania S.Asia C.Asia N.Afr./Mid.East Subsah.Afr." > /n/holylfs05/LABS/jacob_lab/Users/drewpendergrass/CHEEREIO/NRT_methane_paper/2024_07_draft2/FIG8_ramping_stats.txt
#demo: python -u regional_ramping.py -d "/n/holylfs05/LABS/jacob_lab/Users/drewpendergrass/CHEEREIO/METHANE_paper_v02_gammapoint1/postprocess/SNAPSHOT_combined_HEMCO_diagnostics.nc" -type "ensemble" -variable "EmisCH4_Total" -time_factor 31536000 -area_factor 1000000 -mass_factor 1000000000 -med True -y "2019 2020 2021 2022" -fix22 True -global True -o "/n/holylfs05/LABS/jacob_lab/Users/drewpendergrass/CHEEREIO/NRT_methane_paper/2024_07_draft2/FIG8_regional_ramping_widerwindow" -g "2.0x2.5" -l "Canada CONUS Cen.Am. So.Am. Europe Russia China Japan/Koreas SE.Asia Oceania S.Asia C.Asia N.Afr./Mid.East Subsah.Afr." -p "Canada CONUS Cen.Am. So.Am. Europe Russia China Japan/Koreas SE.Asia Oceania S.Asia C.Asia N.Afr./Mid.East Subsah.Afr." -period "2022-07-01 2022-09-15"  > /n/holylfs05/LABS/jacob_lab/Users/drewpendergrass/CHEEREIO/NRT_methane_paper/2024_07_draft2/FIG8_ramping_stats_widerwindow.txt

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import argparse
from scipy.stats import linregress
import pandas as pd
from workshop_tools import *

parser = argparse.ArgumentParser(description='Plots masks.')
parser.add_argument('-d', '--data_to_ramp', type=str, help='Data to calculate ramp. If more than one supplied, take the average.')
parser.add_argument('-type', '--file_type', type=str, help='Does data files contain "control" or "ensemble" data?')
parser.add_argument('-variable', '--variableNames', type=str, help='Name of variable to plot (e.g. EmisCH4_Wetlands). If combining variables, use + symbol to add together (or - to subtract TWO entries only).')
parser.add_argument('-time_factor', '--conversion_time_factor', type=float, default = 1.0, help='Conversion factor applied to data to convert time units.')
parser.add_argument('-area_factor', '--conversion_area_factor', type=float, default = 1.0, help='Conversion factor applied to data to convert area units.')
parser.add_argument('-mass_factor', '--conversion_mass_factor', type=float, default = 1.0, help='Conversion factor applied to data to convert mass units.')
parser.add_argument('-med', '--useMedian', type=str2bool, default=False, help='If true, use median across ensemble. If false, use mean.')
parser.add_argument('-g', '--grid_label', type=str, help='Grid label for mask')
parser.add_argument('-l', '--mask_labels', type=str, help='Mask labels, one for each mask supplied. Space delimited')
parser.add_argument('-p', '--paths_to_masks', type=str, help='Paths to masks. Space delimited.')
parser.add_argument('-y', '--years_for_trend', type=str, help='Years to use to calculate trend, space delimited (e.g. 2019 2020 2021 2022).')
parser.add_argument('-o', '--file_out', type=str, help='Image file produced (no png).')
parser.add_argument('-fix22', '--fix22', type=str2bool, default=True, help='Bias correct 2022 with missing emissions.')
parser.add_argument('-fix23', '--fix23', type=str2bool, default=True, help='Bias correct 2023 with missing emissions.')
parser.add_argument('-global', '--print_global_trend', type=str2bool, default=True, help='Calculate and print (not plot) global trend.')
parser.add_argument('-per22', '--fix22_period_excluded', type=str, default="2022-07-26 2022-08-23", help='Period used to bias correct 2022 (removed and data outside period used for calculation).')
parser.add_argument('-per23', '--fix23_period_excluded', type=str, default="2023-08-16 2023-09-10", help='Period used to bias correct 2023 (removed and data outside period used for calculation).')
parser.add_argument('-cmax', '--cmax_and_cmin_of_slope_plot', type=str, default="", help='cmax and cmin (-1*cmax) of slope plot. Calculated from input data if not supplied')

args = parser.parse_args()

file = args.data_to_ramp.split()
if len(file)>1:
	avgruns = True
else:
	avgruns=False
	file = file[0]
mlabels = args.mask_labels.split()
mpaths_raw = args.paths_to_masks.split()
mpaths = []
budget_folder="/n/holylfs05/LABS/jacob_lab/Users/drewpendergrass/CHEEREIO/NRT_methane_paper"
print_global_trend = args.print_global_trend
mapper = {"Canada":f"{budget_folder}/canada_and_alaska_2x2p5_mask.npy", "CONUS":f"{budget_folder}/conus_2x2p5_mask.npy", "Cen.Am.":f"{budget_folder}/central_america_and_caribbean_2x2p5_mask.npy", "So.Am.":f"{budget_folder}/southamerica_2x2p5_mask.npy", "Europe":f"{budget_folder}/europe_full_no_russia_2x2p5_mask.npy", "Russia":f"{budget_folder}/russia_2x2p5_mask.npy", "China":f"{budget_folder}/china_2x2p5_mask.npy", "Japan/Koreas":f"{budget_folder}/japan_and_korean_peninsula_2x2p5_mask.npy", "SE.Asia":f"{budget_folder}/southeast_asia_2x2p5_mask.npy", "Oceania":f"{budget_folder}/oceania_2x2p5_mask.npy", "S.Asia":f"{budget_folder}/south_asia_2x2p5_mask.npy", "C.Asia":f"{budget_folder}/central_asia_2x2p5_mask.npy", "N.Afr./Mid.East":f"{budget_folder}/africa_north_and_middle_east_2x2p5_mask.npy", "Subsah.Afr.":f"{budget_folder}/africa_subsaharan_2x2p5_mask.npy"}
for m in mpaths_raw:
	if m in mapper:
		mpaths.append(mapper[m])
	else:
		mpaths.append(m)

if print_global_trend:
	mlabels.append("Global")
	mpaths.append(None)

file_out = args.file_out
gridlabel = args.grid_label
fix22=args.fix22
fix23=args.fix23
years_for_trend = [int(y) for y in args.years_for_trend.split()]
variable = args.variableNames
file_type = args.file_type
time_factor = args.conversion_time_factor
area_factor = args.conversion_area_factor
mass_factor = args.conversion_mass_factor
useMedian = args.useMedian
fix22_period_excluded = args.fix22_period_excluded.split()
fix23_period_excluded = args.fix23_period_excluded.split()
cmax = args.cmax_and_cmin_of_slope_plot


conversion_factor = time_factor*area_factor/mass_factor

if avgruns:
	data = [xr.load_dataset(f) for f in file]
else:
	data = xr.load_dataset(file)

lon,lat = getLonLatFromLabel(gridlabel)
lon2d, lat2d = np.meshgrid(lon,lat)

to_plot = np.zeros(lon2d.shape).astype(float)
to_plot_p_value = np.zeros(lon2d.shape).astype(float)
labels = {}

for m,l in zip(mpaths,mlabels):
	if m is not None:
		mask = np.load(m)
	else:
		mask = None
	print('')
	print(f'BEGIN {l}')
	emis_list = []
	for y in years_for_trend:
		adjustyear = False
		if ((y == 2022) and fix22) or ((y == 2023) and fix23):
			adjustyear = True
			if y==2022:
				period_excluded=fix22_period_excluded
			elif y==2023:
				period_excluded=fix23_period_excluded
			timeslice = slice(f"2021-01-01", f"{y}-12-31")
		else:
			timeslice = slice(f"{y}-01-01", f"{y}-12-31")
		if avgruns:
			tempdata_array = [d.sel(time=timeslice) for d in data]
			if '-' in variable:
				vs  = variable.split('-')
			else:
				vs = variable.split('+')
			#Average variables across simulations that will be used for calcualtion in the next step.
			tempdata = tempdata_array[0].copy()
			for v in vs:
				var_array = [d[v] for d in tempdata_array]
				tempdata[v] = xr.concat(var_array, pd.Index(np.arange(0,len(var_array)), name='simulation')).mean(dim='simulation')
		else:
			tempdata = data.sel(time=timeslice)
		if '-' in variable:
			vs = variable.split('-')
			if not adjustyear:
				emis2d = get2DEmissions(tempdata,file_type,vs[0],"avg",conversion_factor,mask=mask,getSpread=True)-get2DEmissions(tempdata,file_type,vs[1],'avg',conversion_factor,mask=mask,getSpread=True)
			else:
				emists1,emists_time = getEmisTS(tempdata,file_type,vs[0],overall_factor=conversion_factor,area_correction=area_factor,mask=mask,getSpread=True)
				emists2,_ = getEmisTS(tempdata,file_type,vs[1],overall_factor=conversion_factor,area_correction=area_factor,mask=mask,getSpread=True)
				emists = emists1-emists2
		else:
			vs = variable.split('+')
			if not adjustyear:
				emis2d = sum([get2DEmissions(tempdata,file_type,v,"avg",conversion_factor,mask=mask,getSpread=True) for v in vs])
			else:
				emists_list = []
				for v in vs:
					emiststemp,emists_time = getEmisTS(tempdata,file_type,v,overall_factor=conversion_factor,area_correction=area_factor,mask=mask,getSpread=True)
					emists_list.append(emiststemp)
				emists = sum(emists_list)
		if adjustyear:
			emis = adjustAnnualEmis(emists,emists_time.data,baseyear = 2021,fixyear = y,period_excluded=period_excluded)
		else:
			emis = getTotalEmis(emis2d,tempdata,area_factor) #1D array of dimension ensemble if of type ensemble
		if file_type == 'ensemble':
			if useMedian:
				emis = np.median(emis)
			else:
				emis = np.mean(emis)
		emis_list.append(emis)
		print(f'{l} {y}: {np.round(emis,2)}')
	slope, intercept, r, p, se = linregress(years_for_trend, emis_list)
	print(f'{l}: slope {np.round(slope,3)} and p value {np.round(p,3)}')
	if mask is not None:
		to_plot += mask*slope
		to_plot_p_value += mask*p

m = Basemap(projection='cyl', resolution='l',llcrnrlat=-90, urcrnrlat=90,llcrnrlon=-180, urcrnrlon=180)
fig = plt.figure(figsize=(10, 10))
m.readshapefile('./WB_countries_Admin0_10m/WB_countries_Admin0_10m', 'WB_countries_Admin0_10m')
if len(cmax)==0:
	vmax = np.max(np.abs(to_plot))
else:
	vmax = float(cmax)
vmin = -1*vmax
mesh = m.pcolormesh(lon2d, lat2d, to_plot,latlon=True,cmap=plt.cm.seismic,vmin=vmin,vmax=vmax)
plt.colorbar()
fig.savefig(f'{file_out}_slope.png')
plt.close(fig)

fig = plt.figure(figsize=(10, 10))
m.readshapefile('./WB_countries_Admin0_10m/WB_countries_Admin0_10m', 'WB_countries_Admin0_10m')
mesh = m.pcolormesh(lon2d, lat2d, to_plot_p_value,latlon=True,cmap=plt.cm.plasma,vmin=0)
plt.colorbar()
fig.savefig(f'{file_out}_pvalue.png')
plt.close(fig)
