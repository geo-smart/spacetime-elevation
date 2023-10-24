import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

plt.style.use('dark_background')
# plt.style.use('default')

def bounds2polygon(xmin, ymin, xmax, ymax, crs):
    """
    Function to return rectangle polygon as GeoDataFrame
    """

    vertices = [(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]
    polygon = Polygon(vertices)
    polygon_gdf = gpd.GeoDataFrame(
        gpd.GeoSeries(polygon), columns=["geometry"], crs=crs
    )
    return polygon_gdf

def calc_percent_coverage(ds, variable = 'band1'):
    total = ds[variable].x.count().values * ds[variable].y.count().values
    coverage = np.count_nonzero(np.isfinite(ds[variable].values), axis=(1,2))
    return coverage / total


def plot_percent_area_coverage(ds, 
                               title = 'Percent area coverage',
                               variable = 'band1', ):
    
    percent_coverage = calc_percent_coverage(ds,variable=variable)
    
    fig,ax = plt.subplots(figsize = (10,2))
    ax.vlines(ds.time.values, ymin=0, ymax=percent_coverage, alpha=0.2, linewidth=10)
    ax.set_ylim([0, 1])
    ax.set_ylabel('%')
    ax.set_title(title)
    
    