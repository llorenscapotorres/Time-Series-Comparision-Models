import numpy as np

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
