import matplotlib.pyplot as plt
import pandas as pd
pd.set_option("display.max_rows", None)
import xarray as xr
import geopandas as gpd

import numpy as np
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
import os



from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


classifiers = {
        'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'knn': KNeighborsClassifier(),
        'svm': SVC(random_state=42),
        'naive_bayes': GaussianNB(),
    }

param_grids_classifier = {
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


regressors = {
        'random_forest': RandomForestRegressor(random_state=42),
        'svr': SVR(),
        'gradient_boosting': GradientBoostingRegressor(random_state=42),
        'linear_regression': LinearRegression(),
        'knn': KNeighborsRegressor(),
    }

param_grids_regressor = {
    'random_forest': {
        'model__n_estimators': [100, 300, 500],
        'model__max_depth': [6, 10, 15],
    },
    'svr': {
        'model__kernel': ['poly', 'linear'],
        'model__C': [0.1, 1, 10],
        'model__degree': [2, 3],  # For poly kernel
    },
    'gradient_boosting': {
        'model__n_estimators': [100, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
    },
    'linear_regression': {
        # No hyperparameters to tune for LinearRegression
    },
    'knn': {
        'model__n_neighbors': [3, 5, 7],
        'model__weights': ['uniform', 'distance'],
    },
}


########################################################################

def cross_validate(train_data, model_class, param_grid, num_fold=5, metric='neg_mean_squared_error'):
    X_train, y_train = train_data
    rkf = RepeatedKFold(n_splits=num_fold, n_repeats=2, random_state=42)
    best_model = None
    best_score = -float('inf')  # Initialize based on the metric
    best_params = None
    mse_scores = []
    r2_scores = []

    for train_index, valid_index in rkf.split(X_train):
        # Split into training and validation sets
        X_train_fold, X_valid_fold = X_train[train_index], X_train[valid_index]
        y_train_fold, y_valid_fold = y_train[train_index], y_train[valid_index]
        
        # Initialize a fresh model for each fold (pipeline + grid search)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model_class)
        ])
        
        # Perform grid search on the training fold
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring=metric)
        grid_search.fit(X_train_fold, y_train_fold)
        
        # Make predictions on the validation fold
        y_pred = grid_search.predict(X_valid_fold)

        if metric == 'accuracy':
            score = accuracy_score(y_valid_fold, y_pred)
        else:
            # Calculate both R² and MSE for regression tasks
            r2 = r2_score(y_valid_fold, y_pred)
            mse = mean_squared_error(y_valid_fold, y_pred)
            r2_scores.append(r2)
            mse_scores.append(mse)

        # Track the best model based on the metric (R² in this case)
        if metric == 'accuracy':
            if score > best_score:
                best_score = score
                best_model = grid_search.best_estimator_  # Best model for this fold
                best_params = grid_search.best_params_
        else:
            # Track the best model based on R²
            if r2 > best_score:
                best_score = r2
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_

    # Print results
    if metric == 'accuracy':
        print(f"Average accuracy: {sum(r2_scores) / len(r2_scores)}")
        print(f"Best accuracy score: {best_score}")
    else:
        # Print both MSE and R² results
        print(f"Average R²: {sum(r2_scores) / len(r2_scores)}")
        print(f"Average MSE: {sum(mse_scores) / len(mse_scores)}")
        print(f"Best R²: {best_score}")
    
    # Return the best model and its parameters
    return best_model, best_params

#####################################################

