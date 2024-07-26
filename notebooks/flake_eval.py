import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.model_selection import RandomizedSearchCV
from typing import Union, Tuple, Dict
from IPython.display import clear_output


df = pd.read_csv('../data/house-prices-advanced-regression-techniques/train.csv')
df_t = pd.read_csv('../data/house-prices-advanced-regression-techniques/test.csv')

ScalerType = Union[StandardScaler, MinMaxScaler]

# Constants
MODEL_PATH = '../models/'
DATA_PATH = '../data/house-prices-advanced-regression-techniques/'
THRESHOLD_NULL = 0.5
THRESHOLD_COUNT = 0.9


def dropping_null_columns(data: pd.DataFrame, threshold: float = THRESHOLD_NULL) -> pd.DataFrame:
    data = data.drop(columns=['Id'])
    columns_to_drop = [col for col in data.columns if data[col].isna().mean() >= threshold]
    return data.drop(columns=columns_to_drop)


def dropping_unnecessary_columns(data: pd.DataFrame, threshold: float = THRESHOLD_COUNT) -> pd.DataFrame:
    columns_to_drop = [col for col in data.columns if data[col].value_counts().max() > threshold * len(data)]
    return data.drop(columns=columns_to_drop)


def ordinal_encoding(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ordinal = OrdinalEncoder()
    categorical_columns = X_train.select_dtypes(include=['object']).columns

    ordinal.fit(X_train[categorical_columns])
    X_train[categorical_columns] = ordinal.transform(X_train[categorical_columns])
    X_test[categorical_columns] = ordinal.transform(X_test[categorical_columns])

    dump(ordinal, MODEL_PATH + 'ordinal_encoder.pkl')
    dump(X.columns, MODEL_PATH + 'columns.pkl')

    return X_train, X_test, y_train, y_test


def standardizing(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler: ScalerType) -> Tuple[np.ndarray, np.ndarray]:
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    dump(scaler, MODEL_PATH + 'standard_scaler.pkl')

    return X_train, X_test


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


def tuning_eval(model: xgb.XGBRegressor, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    y_pred_test_hpt = model.predict(X_test)
    return {
        'R2': round(r2_score(y_test, y_pred_test_hpt), 2),
        'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred_test_hpt)), 2),
        'rmsle': compute_rmsle(y_test, y_pred_test_hpt)
    }


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


def encode_and_update(data: pd.DataFrame, ordinal_path: str) -> OrdinalEncoder:
    ordinal = load(ordinal_path)
    categorical_columns = data.select_dtypes(include=['object']).columns

    for index, col in enumerate(categorical_columns):
        unique_items = set(data[col])
        known_items = set(ordinal.categories_[index])
        new_items = unique_items - known_items

        if new_items:
            ordinal.categories_[index] = np.append(ordinal.categories_[index], list(new_items))

    dump(ordinal, MODEL_PATH + 'ordinal_encoder.pkl')
    return ordinal


def load_pkls() -> Tuple[pd.Index, ScalerType, xgb.XGBRegressor]:
    cols = load(MODEL_PATH + 'columns.pkl')
    standard = load(MODEL_PATH + 'standard_scaler.pkl')
    model = load(MODEL_PATH + 'model.joblib')
    return cols, standard, model


def make_prediction_before_tuning(data: pd.DataFrame) -> str:
    cols, standard, model = load_pkls()
    ids = data['Id']
    df_test = data[cols]
    ordinal = encode_and_update(df_test, MODEL_PATH + 'ordinal_encoder.pkl')

    df_test[df_test.select_dtypes(include=['object']).columns] = ordinal.transform(
        df_test.select_dtypes(include=['object']))
    df_test = standard.transform(df_test)

    y_pred = model.predict(df_test)
    submission_df = pd.DataFrame({'Id': ids, 'SalePrice': y_pred})
    submission_df.to_csv(DATA_PATH + 'submission.csv', index=False)

    return "Submission file created successfully."


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


def build_model(data: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    model, X_train, X_test, y_train, y_test = build_model_before_tuning(data)

    return ({'model performance before tuning': evaluation(model, X_test, y_test)},
            tuning(model, X_train, X_test, y_train, y_test))


m = build_model_before_tuning(df)
print(m)

e = make_prediction_before_tuning(df_t)

# print(m)
# print(e)
