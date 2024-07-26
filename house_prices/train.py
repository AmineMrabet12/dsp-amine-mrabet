import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.model_selection import RandomizedSearchCV
from IPython.display import clear_output
from typing import Tuple, Dict
from house_prices.preprocess import dropping_null_columns, dropping_unnecessary_columns, ordinal_encoding, standardizing


MODEL_PATH = '../models/'


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def evaluation(model: xgb.XGBRegressor, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    y_pred_test = model.predict(X_test)
    return {
        'R2': round(r2_score(y_test, y_pred_test), 2),
        'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 2),
        'rmsle': compute_rmsle(y_test, y_pred_test)
    }


def tuning(model: xgb.XGBRegressor, X_train: np.ndarray,
           X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    params = {
        'n_estimators': [100, 150, 200, 250, 300, 350, 400, 500, 750, 800, 850],
        'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.25, 0.5],
        'min_child_weight': [1, 5, 7, 10],
        'gamma': [0.1, 0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
        'max_depth': [3, 4, 5, 10, 12, 15, 20]
    }
    folds = 5
    param_comb = 100

    random_search = RandomizedSearchCV(model, param_distributions=params,
                                       n_iter=param_comb, n_jobs=-1,
                                       cv=folds, verbose=3,
                                       random_state=42)
    random_search.fit(X_test, y_test)

    best_params = random_search.best_params_
    xgb_reg_hpt = xgb.XGBRegressor(
        subsample=best_params['subsample'],
        n_estimators=best_params['n_estimators'],
        min_child_weight=best_params['min_child_weight'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        gamma=best_params['gamma'],
        colsample_bytree=best_params['colsample_bytree'],
        n_jobs=-1
    )
    xgb_reg_hpt.fit(X_train, y_train)
    clear_output(wait=False)
    dump(xgb_reg_hpt, MODEL_PATH + 'model.joblib')

    return {'model performance after tuning': evaluation(xgb_reg_hpt, X_test, y_test)}


def build_model_before_tuning(data: pd.DataFrame) -> Tuple[xgb.XGBRegressor,
                                                           np.ndarray, np.ndarray,
                                                           np.ndarray, np.ndarray]:
    data = dropping_null_columns(data)
    data = dropping_unnecessary_columns(data)
    X = data.drop(columns=['SalePrice'])
    y = data['SalePrice']
    X_train, X_test, y_train, y_test = ordinal_encoding(X, y)
    X_train, X_test = standardizing(X_train, X_test, StandardScaler())

    xgb_reg = xgb.XGBRegressor(n_jobs=-1)
    xgb_reg.fit(X_train, y_train)
    dump(xgb_reg, MODEL_PATH + 'model_before_tuning.joblib')

    return xgb_reg, X_train, X_test, y_train, y_test


def build_model(data: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    model, X_train, X_test, y_train, y_test = build_model_before_tuning(data)

    return ({'model performance before tuning': evaluation(model, X_test, y_test)},
            tuning(model, X_train, X_test, y_train, y_test))
