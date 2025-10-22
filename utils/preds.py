import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from typing import List

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

def evaluate_preds_global(y_true, y_pred, horizon=None):
    """
    Take in the model predictions and truth values and return evaluation metrics.
    """
    # Convert to numpy arrays with float32
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    if y_true.shape != y_pred.shape:
        raise ValueError(f'Shapes must match, got {y_true.shape} vs {y_pred.shape}')

    # Flatten both arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

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

def evaluate_preds_marginal(y_true, y_pred):
    """
    Calculates the MAE, MSE, RMSE, and MAPE metrics per forecast horizon.
    """
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes must match, got {y_true.shape} vs {y_pred.shape}")
    
    H = y_true.shape[1]  # number of forecast horizons
    results = {}

    for h in range(H):
        mae = np.mean(np.abs(y_true[:, h] - y_pred[:, h]))
        mse = np.mean((y_true[:, h] - y_pred[:, h]) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true[:, h] - y_pred[:, h]) / (y_true[:, h] + 1e-8))) * 100

        results[f'h{h+1}'] = {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}

    return results

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
                                    test_exog_df: pd.DataFrame,
                                    length_prediction: int,
                                    y_true: np.array,
                                    model):
    preds_tst = []
    df_current = train_df.copy()

    for i in range(length_prediction):

        current_exog = test_exog_df.iloc[i:i+1].copy()
        pred = model.predict(
            df=df_current,
            futr_df=current_exog
        )
        preds_tst.append(pred)

        new_row = current_exog.copy()
        new_row['y'] = y_true[i]
        df_current = pd.concat([df_current, new_row], ignore_index=True)

    return [df.iloc[0, 2] for df in preds_tst]

def make_predictions_darts_horizon(ts_train: TimeSeries, 
                                       ts_test: TimeSeries, 
                                       ts_exogenus_train: TimeSeries, 
                                       ts_exogenus_test: TimeSeries,
                                       scaled: bool, 
                                       scaler: Scaler,
                                       horizon: int, 
                                       model):
    current_series = ts_train
    current_exogenus = ts_exogenus_train
    preds = []

    exo_values = ts_exogenus_test.values() if ts_exogenus_test is not None else [None] * len(ts_test)
    limit = len(ts_test) - horizon
    for i, (value, value_exo) in enumerate(zip(ts_test.values(), exo_values)):

        if i >= limit:
            break

        pred = model.predict(n=horizon, series=current_series,
                             past_covariates = current_exogenus if ts_exogenus_test is not None else None)
        if scaled:
            pred = scaler.inverse_transform(pred)
        preds.append(pred.values())

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

def make_predictions_rolling_horizon_nf(train_df: pd.DataFrame,
                                     test_exog_df: pd.DataFrame,
                                     length_prediction: int,
                                     y_true: np.ndarray,
                                     model,
                                     horizon: int = 7) -> np.ndarray:
    """
    Rolling predictions en bloques de tamaño `horizon`.

    Parámetros
    ----------
    train_df : pd.DataFrame
        Dataframe de entrenamiento inicial que contiene columnas exógenas y 'y' hasta el último dato conocido.
    test_exog_df : pd.DataFrame
        Dataframe con las variables exógenas del periodo de test (longitud >= length_prediction).
        Se asume que la fila 0 de test_exog_df corresponde al primer día a predecir.
    length_prediction : int
        Número total de días en el conjunto de test (ej: 127).
    y_true : np.ndarray
        Vector con los valores reales del conjunto de test (longitud >= length_prediction).
    model :
        Modelo que implementa `.predict(df=df_current, futr_df=futr_df)` y devuelve
        una estructura desde la que se pueda extraer las predicciones (aquí se asume DataFrame/array).
    horizon : int
        Horizonte de predicción (por defecto 7).

    Retorna
    -------
    np.ndarray
        Matriz de forma (num_iteraciones, horizon) con las predicciones. num_iteraciones = length_prediction - horizon.
    """

    preds_blocks: List[np.ndarray] = []
    df_current = train_df.copy()

    # número de bloques que quieres generar (según tu requerimiento)
    num_iters = length_prediction - horizon
    if num_iters <= 0:
        raise ValueError("length_prediction debe ser mayor que horizon")

    # Pre-computamos la "última exógena conocida" para cada i durante las iteraciones
    # (test_exog_df debe tener al menos length_prediction filas)
    for i in range(0, num_iters):
        # Índices de los días que queremos predecir: i .. i+horizon-1
        futr = test_exog_df.iloc[i:i + horizon].copy()

        # Disponibilidad de exógenas: solo conocemos exógenas hasta el día i (incluido).
        # Por tanto, para las filas j>i dentro del bloque, rellenamos con la última exógena conocida (fila i).
        if len(futr) < horizon:
            # Si no hay suficientes filas (precaución), rellenamos con la última fila disponible
            last_known = test_exog_df.iloc[i:i+1].reindex(range(horizon)).ffill().iloc[0]
            futr = pd.DataFrame([last_known.copy() for _ in range(horizon)], columns=last_known.index)
        else:
            # Las filas futr.iloc[1:], futr.iloc[2:], ... corresponden a exógenas futuras no observadas.
            # Rellenamos esas filas con la fila futr.iloc[0] (última exógena observada).
            if horizon > 1:
                first_row = futr.iloc[0]
                # Reemplazamos todas las filas j>0 por first_row
                for j in range(1, horizon):
                    futr.iloc[j] = first_row

        # Llamada al modelo (ajusta si tu modelo devuelve otro tipo)
        pred_block = model.predict(df=df_current, futr_df=futr)

        # Extrae las predicciones del bloque en formato 1D numpy array:
        # Aquí asumo que `pred_block` es un DataFrame con la columna 'y' o un array con shape (horizon,)
        if isinstance(pred_block, pd.DataFrame):
            # intentamos extraer la columna 'y' o la tercera columna como en tu código original
            if 'y' in pred_block.columns:
                block_values = pred_block['y'].to_numpy().reshape(-1)[:horizon]
            else:
                # fallback: primera columna
                block_values = pred_block.iloc[:, 0].to_numpy().reshape(-1)[:horizon]
        else:
            # si es numpy array
            block_values = np.asarray(pred_block).reshape(-1)[:horizon]

        preds_blocks.append(block_values)

        # **Actualizamos df_current**: solo añadimos la fila real del primer día del bloque (i)
        new_row = test_exog_df.iloc[i:i+1].copy()
        new_row = new_row.reset_index(drop=True)
        new_row['y'] = y_true[i]
        df_current = pd.concat([df_current, new_row], ignore_index=True)

    return np.vstack(preds_blocks)  # shape (num_iters, horizon)