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
        product="sentinel1_grd_gamma0_20m",
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
    native_crs = notebook_utils.mostcommon_crs(dc, query)
    print(f'Most common native CRS: {native_crs}')
    
    # measurements = ['red','green', 'blue', 'nir', 'scl']
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


def load_sen1(name_vh, name_vv):
    dsvv = rioxarray.open_rasterio(name_vv)
    dsvh = rioxarray.open_rasterio(name_vh)
    return dsvh, dsvv


def create_dataset(train, average_ndvi, dsvh, dsvv):
    loaded_datasets = {}
    for idx, point in train.iterrows():
        key = f"point_{idx + 1}"
        try:
            ndvi_data = average_ndvi.sel(x=point.geometry.x, y=point.geometry.y, method='nearest').values
            vh_data = dsvh.sel(x=point.geometry.x, y=point.geometry.y, method='nearest').values
            vv_data = dsvv.sel(x=point.geometry.x, y=point.geometry.y, method='nearest').values
            loaded_datasets[key] = {
                "data": np.concatenate((ndvi_data, vh_data, vv_data)),
                "label": point.HT_code
                                   }
        except Exception as e:
            # loaded_datasets[key] = None
            print(e)
    return loaded_datasets


def split_train_data(train, label_mapping, datasets):
    label_encoder = LabelEncoder()
    # Fit and transform the labels
    labels = train.Hientrang.values
    numeric_labels = label_encoder.fit_transform([label_mapping[label] for label in labels])
    X = []
    x_new = []
    lb_new = []
    for k, v in datasets.items():
        X.append(v)
    for i in range(len(X)):
        if X[i] is not None:
            x_new.append(X[i]["data"])
            lb_new.append(numeric_labels[i])
    X_train, X_test, y_train, y_test= train_test_split(x_new, lb_new, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# def find_best_model(dataset):
#     X_train, X_val, y_train, y_val = dataset
#     # Tạo RandomForestClassifier mặc định để sử dụng làm mô hình ban đầu trong pipeline
#     base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

#     # Tạo pipeline
#     pipeline = Pipeline([
#         # ('imputer', SimpleImputer(strategy='mean')),
#         ('scaler', StandardScaler()),
#         ('classifier', base_model),
#     ])
#     # Thiết lập các tham số bạn muốn tối ưu hóa
#     param_grid = {
#         'classifier__n_estimators': [100, 300, 500, 700, 1000],
#         'classifier__max_depth': [6, 8, 10, 15, 20],
#         'classifier__criterion': ['gini', 'entropy'],
#     }

#     # Sử dụng GridSearchCV để tìm bộ tham số tốt nhất
#     grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
#     grid_search.fit(X_train, y_train)

#     # In ra bộ tham số tốt nhất
#     best_params = grid_search.best_params_
#     print("Best Parameters:", best_params)

#     # Dự đoán trên tập kiểm tra
#     y_pred = grid_search.predict(X_val)

#     # Đánh giá kết quả
#     accuracy = accuracy_score(y_val, y_pred)
#     print(f"Accuracy: {round(accuracy, 2)*100} %")
#     return grid_search

#############################################################################t
def find_best_model(dataset, model_type):
    X_train, y_train = dataset

    # Dictionary of classifiers
    classifiers = {
        'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'knn': KNeighborsClassifier(),
        'svm': SVC(random_state=42),
        'naive_bayes': GaussianNB(),
    }
    if model_type not in regressors:
        raise ValueError(f"Invalid model_type: {model_type}. Available options are: {list(regressors.keys())}")

    # Select the appropriate classifier based on `model_type`
    classifier = classifiers.get(model_type, RandomForestClassifier(random_state=42, n_jobs=-1))

    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier),
    ])

    # Set up parameter grids for different classifiers
    param_grids = {
        'random_forest': {
            'classifier__n_estimators': [100, 300, 500],
            'classifier__max_depth': [6, 10, 15],
            'classifier__criterion': ['gini', 'entropy'],
        },
        'knn': {
            'classifier__n_neighbors': [3, 5, 7],
            'classifier__weights': ['uniform', 'distance'],
        },
        'svm': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf'],
        },
        'naive_bayes': {
            # No hyperparameters to tune for GaussianNB by default
        },
    }
    
    # Get the parameter grid for the chosen classifier
    param_grid = param_grids.get(model_type, param_grids[model_type])
    # Use GridSearchCV to find the best parameters
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    # Print the classifier name
    print(f"Using classifier: {classifier.__class__.__name__}")
    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)
    # Predict on the validation set
    y_pred = grid_search.predict(X_val)
    # Evaluate the result

    return grid_search

######################################

def find_best_regressor(dataset, model_type):
    X_train, y_train = dataset
    # Dictionary of regressors
    regressors = {
        'random_forest': RandomForestRegressor(random_state=42),
        'svr': SVR(),
        'gradient_boosting': GradientBoostingRegressor(random_state=42),
        'linear_regression': LinearRegression(),
        'knn': KNeighborsRegressor(),
    }
    # Select the appropriate regressor based on `model_type`
    if model_type not in regressors:
        raise ValueError(f"Invalid model_type: {model_type}. Available options are: {list(regressors.keys())}")

    regressor = regressors[model_type]
    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', regressor),
    ])
    # Set up parameter grids for different regressors
    param_grids = {
        'random_forest': {
            'regressor__n_estimators': [100, 300, 500],
            'regressor__max_depth': [6, 10, 15],
        },
        'svr': {
            'regressor__kernel': ['poly', 'linear'],
            'regressor__C': [0.1, 1, 10],
            'regressor__degree': [2, 3],  # For poly kernel
        },
        'gradient_boosting': {
            'regressor__n_estimators': [100, 300],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
        },
        'linear_regression': {
            # No hyperparameters to tune for LinearRegression
        },
        'knn': {
            'regressor__n_neighbors': [3, 5, 7],
            'regressor__weights': ['uniform', 'distance'],
        },
    }
    # Fetch the parameter grid for the selected regressor
    param_grid = param_grids.get(model_type, param_grids['random_forest'])

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print(f"Using: {regressor.__class__.__name__}")
    print (f"Best paras: {grid_search.best_params_}")
    # Return the best model and its parameters
    return grid_search
# Example usage:
# best_model, best_params = find_best_regressor((X_train, y_train), 'gradient_boosting')


########################################################################
def cross_validate(train_data, model, num_fold=5):
    X_train, y_train = train_data
    rkf = RepeatedKFold(n_splits=num_fold, n_repeats=2, random_state=42)
    validation_scores = []
    for train_index, valid_index in rkf.split(X_train):
        # Split into training and validation sets
        X_train_fold, X_valid_fold = X_train[train_index], X_train[valid_index]
        y_train_fold, y_valid_fold = y_train[train_index], y_train[valid_index]
        y_pred = model.predict(X_valid_fold)
        score = mean_squared_error(y_valid_fold, y_pred)
        # Store the score
        validation_scores.append(score)
    print(f"Validation scores over splits: {validation_scores}")
    print(f"Mean validation score: {sum(validation_scores) / len(validation_scores)}")


#################



def save_model(name_file, grid_search):
    dir_save_model = "model_train"
    if not os.path.exists(dir_save_model):
        os.mkdir(dir_save_model)
    joblib.dump(grid_search, os.path.join(dir_save_model, name_file))
    print("Done!")
    
    
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