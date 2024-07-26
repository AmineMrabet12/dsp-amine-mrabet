import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import load
import xgboost as xgb
from IPython.display import clear_output
from typing import Tuple, Union
from house_prices.preprocess import encode_and_update


ScalerType = Union[StandardScaler, MinMaxScaler]
MODEL_PATH = '../models/'
DATA_PATH = '../data/'


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


def make_predictions(data: pd.DataFrame) -> pd.DataFrame:
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
    clear_output(wait=False)
    print("Submission file created successfully.")

    return submission_df
