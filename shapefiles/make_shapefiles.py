import os
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import cartopy.crs as ccrs

# define regions
continents = ['Africa', 'South America', 'Oceania'] # Europe is defined below without Russia
indiv_countries = ['China', 'Canada', 'United States of America', 'Russia']
middle_east = ['Saudi Arabia', 'Yemen', 'Oman', 'Iraq', 'Israel', 'Lebanon', 'Jordan',
                  'Syria', 'Turkey', 'Iran', 'United Arab Emirates', 'Kuwait', 'Qatar']
southeast_asia = ['Myanmar', 'Vietnam', 'Laos', 'Thailand', 'Cambodia', 'East Timor', 
                'Malaysia', 'Singapore', 'Indonesia', 'Brunei', 'Philippines']
japan_koreas = ['Japan', 'South Korea', 'North Korea']
india_pakistan = ['India', 'Pakistan']

# load state vector
sv_path = "~/mhe/Global_2019_annual/StateVector.nc"
sv = xr.load_dataset(sv_path)

# make geodataframe of the state vector elements
dlon = np.median(np.diff(sv.lon.values))
dlat = np.median(np.diff(sv.lat.values))
lon = sv.lon.values
lat = sv.lat.values

# grid cell corners lat/lon, only valid elements
X, Y = np.meshgrid(lon, lat)
valid_lon = X.flatten()
valid_lat = Y.flatten()

coords = np.stack(
    [
        np.column_stack([valid_lon - dlon / 2, valid_lat + dlat / 2]),  # top-left
        np.column_stack([valid_lon + dlon / 2, valid_lat + dlat / 2]),  # top-right
        np.column_stack(
            [valid_lon + dlon / 2, valid_lat - dlat / 2]
        ),  # bottom-right
        np.column_stack(
            [valid_lon - dlon / 2, valid_lat - dlat / 2]
        ),  # bottom-left
    ],
    axis=1,
)

# grid cell geometry on lat/lon coordinate system
gdf_grid = gpd.GeoDataFrame(
    geometry=[Polygon(coord) for coord in coords], crs="EPSG:4326"
)

# geodataframe of countries of the world
gdf_world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
crs = ccrs.Robinson()
crs_proj4 = crs.proj4_init
gdf_world_crs = gdf_world.to_crs(epsg=4326)

# process continents
for c in continents:
    gdf_region = gdf_world_crs[gdf_world['continent'] == c].dissolve()
    # save all necessary shapefiles
    c = c.replace(' ', '-')
    gdf_region.to_file(f"regions/{c.lower()}.shp")
    print(f"Saved {c}")

# replace continent column with individual country name for China, Canada, US, Russia
for c in indiv_countries:
    gdf_region = gdf_world_crs[gdf_world['name'] == c]
    gdf_region.loc[gdf_region['name'].isin(indiv_countries), 'continent'] = c
    # save all necessary shapefiles
    c = c.replace(' ', '-')
    gdf_region.to_file(f"regions/{c.lower()}.shp")
    print(f"Saved {c}")

# replace continent column for Japan, S Korea, N Korea
# replace continent column for India, Pakistan
gdf_region = gdf_world_crs.copy()
gdf_region.loc[gdf_region['name'].isin(japan_koreas), 'continent'] = "Japan/Koreas"
gdf_region.loc[gdf_region['name'].isin(india_pakistan), 'continent'] = "India/Pakistan"

gdf_japan_koreas = gdf_region[gdf_region['continent'] == 'Japan/Koreas'].dissolve()
gdf_india_pakistan = gdf_region[gdf_region['continent'] == 'India/Pakistan'].dissolve()

# save all necessary shapefiles
gdf_japan_koreas.to_file("regions/japan-koreas.shp")
print("Saved Japan and Koreas")
gdf_india_pakistan.to_file("regions/india-pakistan.shp")
print("Saved India and Pakistan")
    
# process multi-country regions
gdf_region = gdf_world_crs.copy()
for i, country in enumerate(gdf_region['name']):
    if country in middle_east:
        gdf_region.loc[i, 'continent'] = 'Middle East'
    elif country in southeast_asia:
        gdf_region.loc[i, 'continent'] = 'Southeast Asia'

gdf_middleeast = gdf_region[gdf_region['continent'] == 'Middle East'].dissolve()
gdf_se_asia = gdf_region[gdf_region['continent'] == 'Southeast Asia'].dissolve()

# save all necessary shapefiles
gdf_middleeast.to_file("regions/middle-east.shp")
print("Saved Middle East")
gdf_se_asia.to_file("regions/southeast-asia.shp")
print("Saved Southeast Asia")

gdf_europe = gdf_world_crs[(gdf_world['continent'] == 'Europe') & (gdf_world['name'] != 'Russia')].dissolve()
gdf_europe.to_file("regions/europe.shp")
print("Saved Europe")

# make merged shapefile
shapefiles = [f for f in os.listdir('regions') if '.shp' in f]
gdf_list = []
for shape in shapefiles:
    gdf = gpd.read_file(f"regions/{shape}")
    gdf_list.append(gdf)

gdf_merged = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
print(gdf_merged)
gdf_merged.plot()
plt.savefig('merged.png')
gdf_merged.to_file("merged.shp")