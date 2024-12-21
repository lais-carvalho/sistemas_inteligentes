from typing import Callable, Union

import numpy as np

from src.si.base.model import Model
from src.si.data.dataset import Dataset
from src.si.metrics.rmse import rmse
from src.si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    The algorithm for "KNNRegressor" is similar to "KNNClassifier".
    However, "KNNRegressor" is suitable for regression problems.
    Therefore, it estimates the average value of the k most similar
    examples instead of the most common class.

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use

    Attributes
    ----------
    dataset: np.ndarray
        The training data
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Initialize the KNNRegressor

        Parameters
        ----------
        k: int
            the number of k nearest examples to consider
        distance: Callable
            The distance function to use
        """
        # parameters
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the closest label of the given sample

        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label of

        Returns
        -------
        label: str or int
            The closest label
        """
        # compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the labels of the k nearest neighbors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        return np.mean(k_nearest_neighbors_labels)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """
        predictions = np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)
        return predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        It returns the RMSE of the model on the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        predictions: np.ndarray
            An array with the predictions

        Returns
        -------
        rmse: float
            The RMSE of the model
    """
        return rmse(dataset.y, predictions)