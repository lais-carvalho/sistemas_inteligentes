import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class LassoRegression(Model):
    def __init__(self, l1_penalty: float = 1.0, alpha: float = 0.001, max_iter: int = 1000, patience: int = 5,
                 scale: bool = True, **kwargs):
        """
        Inicializa o modelo de regressão Lasso.

        Parameters
        ----------
        l1_penalty: float
            O parâmetro de regularização L1.
        alpha: float
            A taxa de aprendizado.
        max_iter: int
            O número máximo de iterações.
        patience: int
            O número de iterações sem melhoria antes de interromper o treinamento.
        scale: bool
            Se os dados devem ser escalonados (normalizados).
        """
        super().__init__(**kwargs)
        self.l1_penalty = l1_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale
        self.theta = None
        self.theta_zero = 0
        self.mean = None
        self.std = None
        self.cost_history = []

    def _soft_threshold(self, r_j, lambda_):
        """
        Operador de soft-thresholding para Lasso.

        Parameters
        ----------
        r_j: float
            O resíduo para o coeficiente j.
        lambda_: float
            O parâmetro de regularização L1.

        Returns
        -------
        float
            O valor atualizado de theta_j após a aplicação do soft-thresholding.
        """
        if r_j > lambda_:
            return r_j - lambda_
        elif r_j < -lambda_:
            return r_j + lambda_
        else:
            return 0

    def _fit(self, dataset: Dataset) -> 'LassoRegression':
        """
        Ajusta o modelo ao conjunto de dados

        Parameters
        ----------
        dataset: Dataset
            O dataset para ajustar o modelo

        Returns
        -------
        self: LassoRegression
            O modelo ajustado
        """
        X = dataset.X

        if self.scale:
            # calcular média e desvio padrão
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            # escalonar os dados
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        y = dataset.y
        m, n = dataset.shape()

        # inicializar os parâmetros do modelo
        self.theta = np.zeros(n)
        self.theta_zero = 0
        self.cost_history = []

        early_stopping = 0
        i = 0

        while i < self.max_iter and early_stopping < self.patience:

            previous_theta = self.theta.copy()

            for j in range(n):
                # Calcular o resíduo r_j
                r_j = np.sum(X[:, j] * (
                            y - (self.theta_zero + np.dot(X, self.theta) - X[:, j] * self.theta[j])))

                # Atualizar theta_j com o soft-thresholding
                self.theta[j] = self._soft_threshold(r_j, self.l1_penalty) / np.sum(X[:, j] ** 2)

            # Atualizar o intercepto theta_0
            self.theta_zero = np.mean(y - np.dot(X, self.theta))

            # Calcular o custo
            y_pred = np.dot(X, self.theta) + self.theta_zero
            cost = self._cost(y, y_pred)
            self.cost_history.append(cost)

            # Verificar se houve melhora no custo
            if i > 0 and self.cost_history[i] > self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0

            i += 1

        return self

    def _cost(self, y, y_pred):
        """
        Calcula a função de custo para o modelo.

        Parameters
        ----------
        y: np.array
            Os valores reais.
        y_pred: np.array
            Os valores preditos.

        Returns
        -------
        float
            O custo (erro quadrático médio + regularização L1).
        """
        m = len(y)
        error = np.sum((y_pred - y) ** 2)
        regularization = self.l1_penalty * np.sum(np.abs(self.theta))
        return (error + regularization) / (2 * m)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Faz a predição dos valores de Y.

        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        np.array
            Os valores preditos de Y.
        """
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Compute the Mean Square Error of the model on the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on.

        predictions: np.ndarray
            Predictions made by the model.

        Returns
        -------
        mse: float
            The Mean Square Error of the model.
        """
        return mse(dataset.y, predictions)