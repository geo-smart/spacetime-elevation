import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import folium
from folium.plugins import Draw
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

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
    
    
def select_points_on_map(gdf, 
                         output_file_name = 'points.geojson'):
    
    m = gdf.explore(tiles=None,
                    style_kwds=dict(fillOpacity=0),
                    name="polygon")
    
    folium.TileLayer(
        "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr='Google',
        opacity=0.8,
        name='Google basemap',
    ).add_to(m)
    

    Draw(export=True,
         filename=output_file_name,
         position = 'topleft').add_to(m)
    
    folium.LayerControl(collapsed=True).add_to(m)
    
    minimap = folium.plugins.MiniMap(position="bottomleft",)
    m.add_child(minimap)

    
    return m

def extract_linestring_coords(linestring):
    """
    Function to extract x, y coordinates from linestring object
    Input:
    shapely.geometry.linestring.LineString
    Returns:
    [x: np.array,y: np.array]
    """
    x = []
    y = []
    for coords in linestring.coords:
        x.append(coords[0])
        y.append(coords[1])
    return [np.array(x), np.array(y)]


def get_test_time_series(da, points):
    """
    Extracts time series at x and y coordinates.
    Removes nans and subtracts the mean of each time series
    """
    x_coords, y_coords = points
    dates = np.array([pd.to_datetime(x) for x in da['time'].values])

    tmp = []
    for i, v in enumerate(x_coords):
        sub = da.sel(x=x_coords[i],
                      y=y_coords[i],
                      method="nearest")

        tmp.append(sub.values)

    test_time = []
    test_data = []
    for i in tmp:
        mask = np.isfinite(i)
        test_data.append(i[mask] - i[mask].mean())
        test_time.append(dates[mask])
    return test_data, test_time

def plot_time_series_gallery(
    x_values,
    y_values,
    masked_x_values=None,
    masked_y_values=None,
    titles=None,
    predictions_df_list=None,
    std_df_list=None,
    x_ticks_off=False,
    y_ticks_off=False,
    sharex=True,
    sharey=True,
    xlim = None,
    ylim=None, 
    y_label='Elevation (m)',
    x_label='Time',
    figsize=(15, 10),
    cmap = None,
    legend=True,
    linestyle="none",
    legend_labels=[
        'Obs','Obs filt', 'Obs mean', 'Prediction', 'Prediction STD',
    ],
    random_choice = False,
    output_file = None,
):

    if cmap:
        cmap_colors = []
        norm = matplotlib.colors.Normalize(vmin=0, vmax=len(x_values))
        if isinstance(cmap, type('')):
            cmap = colormaps.get_cmap(cmap)

    rows, columns = get_row_column(len(x_values))

    fig = plt.figure(figsize=figsize)
    axes = []
    for i in range(rows * columns):
        try:
            if cmap:
                c = cmap(norm(i))
            else:
                c = 'b'
            x, y = x_values[i], y_values[i]
            ax = plt.subplot(rows, columns, i + 1, aspect="auto")
            ax.plot(x, y, marker="o", c=c, linestyle=linestyle, label=legend_labels[0])
            if masked_x_values:
                fx,fy = masked_x_values[i], masked_y_values[i]
                ax.plot(fx, fy, marker="o", c="r", linestyle=linestyle, label=legend_labels[1])
            if random_choice:
                random_index = np.random.choice(np.arange(x.size))
                ax.plot(x[random_index], y[random_index], marker="o", c="r", linestyle=linestyle)
            if x_ticks_off:
                ax.set_xticks(())
            if y_ticks_off:
                ax.set_yticks(())

            # ax.axhline(np.mean(y_values[i]),color='k',alpha=0.2, label=legend_labels[2])
            axes.append(ax)
            
        except:
            pass
    if not isinstance(predictions_df_list, type(None)):
        for idx, df in enumerate(predictions_df_list):
            try:
                std_df = std_df_list[idx]
            except:
                std_df = None

            for i, series in df.items():
                ax = axes[i]
                try:
                    series.plot(ax=ax, c="C" + str(idx + 1), label=legend_labels[idx + 3])
                except:
                    series.plot(ax=ax, c="C" + str(idx + 1), label=legend_labels[idx + 1])
                if not isinstance(std_df, type(None)):
                    x = series.index.values
                    y = series.values
                    std_prediction = std_df[i].values
                    ax.fill_between(
                        x,
                        y - 1.96 * std_prediction,
                        y + 1.96 * std_prediction,
                        alpha=0.2,
                        label=legend_labels[idx + 4],
                        color="C" + str(idx + 1),
                    )

    if titles:
        for i, ax in enumerate(axes):
            ax.set_title(titles[i])

    if legend:
        axes[0].legend(loc='lower left')
        
    if y_label:
        for ax in axes:
            if ax in axes[::columns]:
                ax.set_ylabel(y_label)
                
    if x_label:
        for ax in axes[-columns:]:
            ax.set_xlabel(x_label)
                
    if sharex:
        mins = []
        maxs = []
        for ax in axes:
            xmin, xmax = ax.get_xlim()
            mins.append(xmin)
            maxs.append(xmax)
        xmin = min(mins)
        xmax = max(maxs)
        for ax in axes:
            if ax in axes[-columns:]:
                ax.set_xlim(xmin,xmax)
            else:
                ax.set_xlim(xmin,xmax)
                ax.set_xticks(())
    if sharey:
        mins = []
        maxs = []
        for ax in axes:
            ymin, ymax = ax.get_ylim()
            mins.append(ymin)
            maxs.append(ymax)
            ax.axhline(0,color='k',alpha=0.2)
        ymin = min(mins)
        ymax = max(maxs)
        for ax in axes:
            if ax in axes[::columns]:
                ax.set_ylim(ymin,ymax)
            else:
                ax.set_ylim(ymin,ymax)
                ax.set_yticks(())
    if ylim:
        for ax in axes:
            ax.set_ylim(ylim[0],ylim[1])

    if xlim:
        for ax in axes:
            ax.set_xlim(xlim[0],xlim[1])
            
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        
def xr_plot_count_std_glacier(
    count_da,
    std_da,
    glacier_gdf=None,
    flowline_gdf=None,
    points=None,
    points_cmap=None,
    plot_to_glacier_extent=False,
    count_vmin=1,
    count_vmax=50,
    count_cmap="gnuplot",
    std_vmin=0,
    std_vmax=20,
    std_cmap="cividis",
    alpha=None,
    ticks_off=False,
    output_file = None,
):
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    ax = axes[0]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cmap = plt.cm.get_cmap(count_cmap, count_vmax)
    norm = matplotlib.colors.Normalize(vmin=count_vmin, vmax=count_vmax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, extend="max", alpha=alpha)
    cbar.set_label(label="DEM count", size=12)
    count_da.plot(ax=ax, cmap=cmap, add_colorbar=False, alpha=alpha, vmin=count_vmin, vmax=count_vmax)

    legend_elements = []
    if isinstance(glacier_gdf, type(gpd.GeoDataFrame())):
        legend_elements.append(Line2D([0], [0], color="k", label="Glacier Outline"))
    if isinstance(flowline_gdf, type(gpd.GeoDataFrame())):
        legend_elements.append(Line2D([0], [0], color="orange", label="Flowlines"))
    if points:
        legend_elements.append(Line2D([0], [0], color="b", label="Observations", marker="o", linestyle="none"))
    if legend_elements:
        ax.legend(handles=legend_elements, loc="best")

    ax = axes[1]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cmap = plt.cm.get_cmap(std_cmap)
    norm = matplotlib.colors.Normalize(vmin=std_vmin, vmax=std_vmax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, extend="max", alpha=alpha)
    cbar.set_label(label="STD [m]", size=12)
    std_da.plot(ax=ax, cmap=cmap, add_colorbar=False, alpha=alpha, vmin=std_vmin, vmax=std_vmax)

    if ticks_off:
        for ax in axes:
            ax.set_xticks(())
            ax.set_yticks(())

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_title("")
        if points:
            if points_cmap:
                ax.scatter(points[0], points[1], marker="o", c=np.arange(len(points[0])), cmap = points_cmap)
            else:
                ax.plot(points[0], points[1], marker="o", color="b", linestyle="none")         
        if isinstance(glacier_gdf, type(gpd.GeoDataFrame())):
            glacier_gdf.plot(ax=ax, facecolor="none", legend=True)
        if isinstance(flowline_gdf, type(gpd.GeoDataFrame())):
            flowline_gdf.plot(ax=ax, color="orange", legend=True)
        if plot_to_glacier_extent:
            glacier_bounds = glacier_gdf.bounds.values[0]
            ax.set_xlim(glacier_bounds[0], glacier_bounds[2])
            ax.set_ylim(glacier_bounds[1], glacier_bounds[3])
#     plt.tight_layout()
    plt.show()
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        plt.close('all')
        
def get_row_column(n):
    """
    returns largest factor pair for int n
    makes rows the larger number
    """
    max_pair = max([(i, n / i) for i in range(1, int(n ** 0.5) + 1) if n % i == 0])
    rows = int(max(max_pair))
    columns = int(min(max_pair))

    # in case n is odd
    # check if you get a smaller pair by adding 1 to make number even
    if not check_if_number_even(n):
        n = make_number_even(n)
        max_pair = max([(i, n / i) for i in range(1, int(n ** 0.5) + 1) if n % i == 0])
        alt_rows = int(max(max_pair))
        alt_columns = int(min(max_pair))

        if (rows, columns) > (alt_rows, alt_columns):
            return (alt_rows, alt_columns)
        else:
            return (rows, columns)
    return (rows, columns)

def check_if_number_even(n):
    """
    checks if int n is an even number
    """
    if (n % 2) == 0:
        return True
    else:
        return False


def make_number_even(n):
    """
    adds 1 to int n if odd number
    """
    if check_if_number_even(n):
        return n
    else:
        return n + 1
    