import numpy as np
from sklearn.metrics import mean_squared_error

def MAPE(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between true and predicted values.

    Parameters:
    - y_test (np.ndarray): Array of true target values.
    - y_pred (np.ndarray): Array of predicted target values.

    Returns:
    - float: MAPE value as a percentage.
    """
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

def MAE(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.

    Parameters:
    - y_test (np.ndarray): Array of true target values.
    - y_pred (np.ndarray): Array of predicted target values.

    Returns:
    - float: MAE value.
    """
    return np.mean(np.abs(y_test - y_pred))

def RMSE(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters:
    - y_test (np.ndarray): Array of true target values.
    - y_pred (np.ndarray): Array of predicted target values.

    Returns:
    - float: RMSE value.
    """
    MSE = mean_squared_error(y_test, y_pred)
    return np.sqrt(MSE)
