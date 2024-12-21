import numpy as np

def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between the true values and predicted values.

    Arguments:
    - y_true: np.ndarray, real values of y.
    - y_pred: np.ndarray, predicted values of y.

    Returns:
    - float: The RMSE value.
    """
    return np.sqrt(((y_true - y_pred) ** 2).mean())