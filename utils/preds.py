import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

def make_preds(model, input_data):
    """
    Uses a model to make predictions on input_data.
    """
    forecast = model.predict(input_data)
    return np.squeeze(forecast)  # return 1D array of predictions

def evaluate_preds(y_true, y_pred):
    """
    Take in the model predictions and truth values and return evaluation metrics.
    """
    # Convert to numpy arrays with float32
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    # Calculate metrics
    mae = np.mean(np.abs(y_true - y_pred))                       # Mean Absolute Error
    mse = np.mean((y_true - y_pred) ** 2)                        # Mean Squared Error
    rmse = np.sqrt(mse)                                          # Root Mean Squared Error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100  # Mean Absolute Percentage Error

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape
    }

def make_predictions_rolling_one_darts(ts_train: TimeSeries, 
                                       ts_test: TimeSeries, 
                                       ts_exogenus_train: TimeSeries, 
                                       ts_exogenus_test: TimeSeries,
                                       scaled: bool, 
                                       scaler: Scaler, 
                                       model):
    current_series = ts_train
    current_exogenus = ts_exogenus_train
    preds = []

    exo_values = ts_exogenus_test.values() if ts_exogenus_test is not None else [None] * len(ts_test)
    for value, value_exo in zip(ts_test.values(), exo_values):

        if ts_exogenus_test is not None:
            next_time_cov = current_exogenus.end_time() + current_exogenus.freq
            value_exo_ts = TimeSeries.from_times_and_values(
                times=pd.DatetimeIndex([next_time_cov], freq=current_exogenus.freq),
                values=value_exo.reshape(1, -1)
            )
            current_exogenus = current_exogenus.append(value_exo_ts)

        pred = model.predict(n=1, series=current_series,
                             future_covariates = current_exogenus if ts_exogenus_test is not None else None)
        if scaled:
            pred = scaler.inverse_transform(pred)
        preds.append(pred.values()[0][0])

        next_time = current_series.end_time() + current_series.freq
        value_ts = TimeSeries.from_times_and_values(
            times=pd.DatetimeIndex([next_time], freq=current_series.freq),
            values=[[value]]
        )
        current_series = current_series.append(value_ts)

    return preds

def make_predictions_rolling_one_past_covariates_darts(ts_train: TimeSeries, 
                                       ts_test: TimeSeries, 
                                       ts_exogenus_train: TimeSeries, 
                                       ts_exogenus_test: TimeSeries,
                                       scaled: bool, 
                                       scaler: Scaler, 
                                       model):
    current_series = ts_train
    current_exogenus = ts_exogenus_train
    preds = []

    exo_values = ts_exogenus_test.values() if ts_exogenus_test is not None else [None] * len(ts_test)
    for value, value_exo in zip(ts_test.values(), exo_values):

        pred = model.predict(n=1, series=current_series,
                             past_covariates = current_exogenus if ts_exogenus_test is not None else None)
        if scaled:
            pred = scaler.inverse_transform(pred)
        preds.append(pred.values()[0][0])

        next_time = current_series.end_time() + current_series.freq
        value_ts = TimeSeries.from_times_and_values(
            times=pd.DatetimeIndex([next_time], freq=current_series.freq),
            values=[[value]]
        )
        current_series = current_series.append(value_ts)

        if ts_exogenus_test is not None:
            next_time_cov = current_exogenus.end_time() + current_exogenus.freq
            value_exo_ts = TimeSeries.from_times_and_values(
                times=pd.DatetimeIndex([next_time_cov], freq=current_exogenus.freq),
                values=value_exo.reshape(1, -1)
            )
            current_exogenus = current_exogenus.append(value_exo_ts)

    return preds

def make_predictions_rolling_one_nf(train_df: pd.DataFrame,
                                    test_df: pd.DataFrame,
                                    length_prediction: int,
                                    y_true: np.array,
                                    model):
    preds_tst = []
    df_current = train_df.copy()

    for i in range(length_prediction):

        current_exog = test_df.iloc[i:i+1].copy()
        pred = model.predict(
            df=df_current,
            futr_df=current_exog
        )
        preds_tst.append(pred)

        new_row = current_exog.copy()
        new_row['y'] = y_true[i]
        df_current = pd.concat([df_current, new_row], ignore_index=True)

    return [df.iloc[0, 2] for df in preds_tst]