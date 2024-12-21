import numpy as np
from unittest import TestCase
from sklearn.metrics import mean_squared_error
from src.si.metrics.rmse import rmse

class TestRMSE(TestCase):

    def test_rmse(self):
        # Dados de exemplo
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])

        # Calcular RMSE com a função personalizada
        custom_rmse = rmse(y_true, y_pred)

        # Calcular RMSE com a função do scikit-learn
        sklearn_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Verificar se o RMSE está dentro de uma margem de erro
        self.assertAlmostEqual(custom_rmse, sklearn_rmse)