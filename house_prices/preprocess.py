import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump
from typing import Union, Tuple


ScalerType = Union[StandardScaler, MinMaxScaler]

MODEL_PATH = '../models/'
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
