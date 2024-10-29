import matplotlib.pyplot as plt

# Common imports and settings
import os, sys
os.environ['USE_PYGEOS'] = '0'
from IPython.display import Markdown
import pandas as pd
pd.set_option("display.max_rows", None)
import xarray as xr
import geopandas as gpd


# Datacube
import datacube
from datacube.utils.rio import configure_s3_access
from datacube.utils import masking
from datacube.utils.cog import write_cog
# https://github.com/GeoscienceAustralia/dea-notebooks/tree/develop/Tools
from dea_tools.plotting import display_map, rgb
from dea_tools.datahandling import mostcommon_crs

# EASI defaults
easinotebooksrepo = '/home/jovyan/easi-notebooks'
if easinotebooksrepo not in sys.path: sys.path.append(easinotebooksrepo)
from easi_tools import EasiDefaults, xarray_object_size, notebook_utils, unset_cachingproxy
from easi_tools.load_s2l2a import load_s2l2a_with_offset
from dask.distributed import progress

# Data tools
import numpy as np
from datetime import datetime

# Datacube
from datacube.utils import masking  # https://github.com/opendatacube/datacube-core/blob/develop/datacube/utils/masking.py
from odc.algo import enum_to_bool   # https://github.com/opendatacube/odc-algo/blob/main/odc/algo/_masking.py
from odc.algo import xr_reproject   # https://github.com/opendatacube/odc-algo/blob/main/odc/algo/_warp.py
from datacube.utils.geometry import GeoBox, box  # https://github.com/opendatacube/datacube-core/blob/develop/datacube/utils/geometry/_base.py

# Holoviews, Datashader and Bokeh
import hvplot.pandas
import hvplot.xarray
import holoviews as hv
import panel as pn
import colorcet as cc
import cartopy.crs as ccrs
from datashader import reductions
from holoviews import opts
# from utils import load_data_geo
import rasterio
import rioxarray
# import geoviews as gv
# from holoviews.operation.datashader import rasterize
hv.extension('bokeh', logo=False)

from deafrica_tools.bandindices import calculate_indices
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from shapely.geometry import Point, Polygon
import geopandas as gpd
from pyproj import CRS
from matplotlib.colors import ListedColormap
from holoviews import opts
from datashader import reductions
from bokeh.models.tickers import FixedTicker
from rioxarray.merge import merge_arrays

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
###
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


def calculate_average(data, time_pattern='1M'):
    return data.resample(time=time_pattern).mean().persist()

def load_data_sen1(dc, date_range, coordinates):
    longtitude_range, latitude_range = coordinates
    data_sen1 = dc.load(
        product="sentinel1_grd_gamma0_10m",
        x=longtitude_range,
        y=latitude_range,
        time=date_range,
        measurements=["vv", "vh"],
        output_crs="EPSG:32648",
        resolution=(-10,10),
        dask_chunks={"x":2048, "y":2048},
        skip_broken_datasets=True,
        group_by='solar_day'
    )
    
    notebook_utils.heading(notebook_utils.xarray_object_size(data_sen1))
    display(data_sen1)
    dsvh = data_sen1.vh
    dsvv = data_sen1.vv
    
    return dsvh, dsvv

def load_data_sen2(dc, date_range, coordinates):
    longtitude_range, latitude_range = coordinates
    product = 's2_l2a'
    query = {
        'product': product,                     # Product name
        'x': longtitude_range,    # "x" axis bounds
        'y': latitude_range,      # "y" axis bounds
        'time': date_range,           # Any parsable date strings
    }
    native_crs = "EPSG:32648"
    measurements = ['red', 'nir', 'scl']
    load_params = {
        'measurements': measurements,                   # Selected measurement or alias names
        'output_crs': native_crs,                       # Target EPSG code
        'resolution': (-10, 10),                        # Target resolution
        'group_by': 'solar_day',                        # Scene grouping
        'dask_chunks': {'x': 2048, 'y': 2048},          # Dask chunks
    }
    data = load_s2l2a_with_offset(
        dc,
        query | load_params   # Combine the two dicts that contain our search and load parameters
    )
    return data

def mask_cloud(data):
    flag_name = 'scl'
    flag_desc = masking.describe_variable_flags(data[flag_name])  # Pandas dataframe
    display(flag_desc.loc['qa'].values[1])
    # Create a "data quality" Mask layer
    flags_def = flag_desc.loc['qa'].values[1]
    good_pixel_flags = [flags_def[str(i)] for i in [2, 4, 5, 6]]  # To pass strings to enum_to_bool()

    # enum_to_bool calculates the pixel-wise "or" of each set of pixels given by good_pixel_flags
    # 1 = good data
    # 0 = "bad" data
    good_pixel_mask = enum_to_bool(data[flag_name], good_pixel_flags)
    data_layer_names = [x for x in data.data_vars if x != 'scl']
    # Apply good pixel mask to blue, green, red and nir.
    result = data[data_layer_names].where(good_pixel_mask).persist()
    return result


def fill_nan(ndvi, time_split):
    rs = []
    for times in time_split:
        tmp = ndvi.sel(time=times)
        fill_ds = tmp.sel(time=times).bfill(dim='time')
        fill_ds = fill_ds.sel(time=times).ffill(dim='time')   
        rs.append(fill_ds)
    merged_ndvi = xr.concat([i for i in rs], dim="time")
    fill_m = merged_ndvi.bfill(dim="time")
    fill_m = fill_m.ffill(dim="time")
    return fill_m



def load_data_geo(path: str):
    gdf = gpd.read_file(path)
    return gdf

#######################################################################
def extract_data_point(train, average_ndvi, dsvh, dsvv):
    loaded_datasets = {}
    for idx, point in train.iterrows():
        key = f"point_{idx + 1}"
        try:
            ndvi_data = average_ndvi.sel(x=point.geometry.x, y=point.geometry.y, method='nearest').values
            vh_data = dsvh.sel(x=point.geometry.x, y=point.geometry.y, method='nearest').values
            vv_data = dsvv.sel(x=point.geometry.x, y=point.geometry.y, method='nearest').values
            loaded_datasets[key] = {
                "data": np.stack((ndvi_data, vh_data, vv_data), axis=0),
                "label": point.HT_code
            }
        except Exception as e:
            # Handle the exception if necessary
            print(f"Error at point {key}: {e}")
    return loaded_datasets
###################################################################
def create_dataset_new(datasets):
    data_points = []
    labels = []
    for point_key, point_data in datasets.items():
        data_array = point_data['data']
        label = point_data['label']
        for row in data_array:
            # Flatten each row and add it as a data point
            flattened_data = data_array.flatten()
            data_points.append(flattened_data)
            # Append the label for each row in the data array
            labels.append(label)
    return data_points, labels
##################################################################
def save_model(name_file, grid_search):
    dir_save_model = "model_train"
    if not os.path.exists(dir_save_model):
        os.mkdir(dir_save_model)
    joblib.dump(grid_search, os.path.join(dir_save_model, name_file))
    print("Done!")
    
######################################################################
regressors = {
        'random_forest': RandomForestRegressor(random_state=42),
        'svr': SVR(),
        'gradient_boosting': GradientBoostingRegressor(random_state=42),
        'linear_regression': LinearRegression(),
        'knn': KNeighborsRegressor(),
    }

param_grids_regressor = {
    'random_forest': {
        'n_estimators': [100, 300, 500, 1000],
        'max_depth': [6, 10, 15],
    },
    'svr': {
        'kernel': ['poly', 'linear'],
        'C': [0.1, 1, 10],
        'degree': [2, 3],  # For poly kernel
    },
    'gradient_boosting': {
        'n_estimators': [100, 300],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5],
    },
    'linear_regression': {
        # 'alpha': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        # 'fit_intercept': [True, False],
        # 'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'],
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 10, 20, 30, 50],
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3],
    },
}


def find_best_regression(x_train, y_train, model, param_grid, cv=5, scoring='neg_mean_squared_error'):
    # Initialize GridSearchCV with the model, parameter grid, and cross-validation settings
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
    # Fit the grid search to the training data
    grid_search.fit(x_train, y_train)
    # Get the best model, parameters, score, and full CV results
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    cv_results = grid_search.cv_results_  # Includes performance metrics for each parameter combination
    
    return best_model, best_params, best_score, cv_results
#############################################################################
classifiers = {
        'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'knn': KNeighborsClassifier(),
        'svm': SVC(random_state=42),
        'naive_bayes': GaussianNB(),
    }

param_grids_classifier = {
    'random_forest': {
        'n_estimators': [100, 300, 500, 700, 1000],
        'max_depth': [6, 8, 10, 15, 20],
        'criterion': ['gini', 'entropy'],
    },
    'knn': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
    },
    'svm': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
    },
    'naive_bayes': {
        # No hyperparameters to tune for GaussianNB by default
    },
}

######################################################################
def find_best_classifier(x_train, y_train, model, param_grid, cv=5, scoring='accuracy'):
    # Initialize GridSearchCV with the model, parameter grid, and cross-validation settings
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
    # Fit the grid search to the training data
    grid_search.fit(x_train, y_train)
    # Get the best model, parameters, score, and full CV results
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    cv_results = grid_search.cv_results_  # Includes performance metrics for each parameter combination
    
    return best_model, best_params, best_score, cv_results










 #########################################################################################   
def predict(model, data_crs, ndvi, vh, vv):
    data_predict = []
    for i in range(ndvi.shape[1]):
        ndvi_tmp = ndvi.isel(y=i).values
        vh_data = vh.sel(y=ndvi.y.values[i], method='nearest').values
        vv_data = vv.sel(y=ndvi.y.values[i], method='nearest').values
        all_tmp = np.concatenate((ndvi_tmp, vh_data, vv_data), axis=0)
        data_predict.extend(all_tmp.T)
    y_pred = model.predict(data_predict)
    final_label = y_pred.reshape(ndvi.y.shape[0], ndvi.x.shape[0])
    
    final_xarray_save = xr.DataArray(final_label, dims=("y", "x"))
    final_xarray_save = final_xarray_save.rio.write_crs(data_crs)

    x_values = ndvi.x.values
    y_values = ndvi.y.values

    data_array = xr.DataArray(final_xarray_save,
                              coords={'x': x_values, 'y': y_values},
                              dims=['y', 'x'])
    data_array = data_array.rio.write_crs(ndvi.rio.crs)
    return data_array


def cut_according_shp(thuanhoa_path, average_ndvi, data_array):
    gdf = gpd.read_file(thuanhoa_path)
    gdf = gdf.to_crs(average_ndvi.rio.crs)
    polygon_coords = list(gdf.geometry.values[0].exterior.coords)
    polygon_coordinates = [(x, y) for x, y in polygon_coords]

    geometries = [
        {
            'type': 'Polygon',
            'coordinates': [polygon_coordinates]
        }
    ]
    region_result = data_array.rio.clip(geometries, data_array.rio.crs, drop=False)
    region_result = region_result.where(region_result >= 0, float('nan'))
    return region_result


def compare(KD_path, KetQuaPhanLoaiDat, CODE_MAP, HT_MAP):
    gdf = gpd.read_file(KD_path, crs="EPSG:9209")
    polygon = gdf.geometry.values
    label = gdf.tenchu.values
    ouput_image = rioxarray.open_rasterio(KetQuaPhanLoaiDat)
    code_tq = HT_MAP["TQ"]["data"][0]
    code_pnn = HT_MAP["PNN"]["data"][0]
    result = {}
    for key, values in HT_MAP.items():
        print(f"process {key}")
        array_list = []
        for i in range(len(polygon)):
            po = polygon[i]
            lb = label[i]
            code_lb = CODE_MAP.get(lb, code_tq)
            try:
                qr = ouput_image.rio.clip([po], "EPSG:9209")
                if code_lb in values["data"]:
                    if code_lb == code_pnn:
                        qr = qr.where((qr != float(code_pnn)), np.nan)
                        # qr = qr.where((qr != 3.0), np.nan)
                    elif code_lb == code_tq:
                        qr = qr.where((qr != float(code_pnn)), np.nan)
                        qr = qr.where((qr != 3.0), np.nan)
                    else: 
                        qr = qr.where(qr != float(code_lb), np.nan)
                else:
                    qr.values[:, :, :] = np.nan
                array_list.append(qr)
            except Exception as e:
               pass
        result.update({key: array_list})
    return result


def save_result(result, save_path, HT_MAP):
    # cmap = ListedColormap(colors)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for k, v in result.items():
        rs = merge_arrays(v, nodata = np.nan)
        rs.rio.to_raster(f"{save_path}/{k}.tif")
        print(f"save {save_path}/{k}.tif")
        # img = rs.plot(cmap=cmap, add_colorbar=False)
        # cbar = plt.colorbar(img)
        # cbar.ax.set_yticklabels(labels)
        # plt.title(f'{HT_MAP[k]["name"]}')
        # plt.axis('off')
        # plt.show()
#################################################       
        