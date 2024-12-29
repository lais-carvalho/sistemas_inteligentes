import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from src.si.metrics.accuracy import accuracy


class CategoricalNB(Model):
    """
    Naive Bayes classifier for categorical features.

    Parameters
    ----------
    smoothing: float, default=1.0
        Laplace smoothing to avoid zero probabilities.

    Attributes
    ----------
    class_prior: np.ndarray
        Prior probabilities for each class.
    feature_probs: np.ndarray
        Probabilities for each feature for each class being present (1).
    """

    def __init__(self, smoothing: float = 1.0, **kwargs):
        """
        Initialize the CategoricalNB model.

        Parameters
        ----------
        smoothing: float, default=1.0
            Laplace smoothing to avoid zero probabilities.
        """
        super().__init__(**kwargs)
        self.smoothing = smoothing
        self.class_prior = None
        self.feature_probs = None

    def _fit(self, dataset: Dataset) -> 'CategoricalNB':
        """
        Fit the model to the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The training dataset.

        Returns
        -------
        self: CategoricalNB
            The fitted model.
        """
        n_samples, n_features = dataset.X.shape
        classes = np.unique(dataset.y)
        n_classes = len(classes)

        # Initialize counts and probabilities
        class_counts = np.zeros(n_classes, dtype=np.float64)
        feature_counts = np.zeros((n_classes, n_features), dtype=np.float64)
        self.class_prior = np.zeros(n_classes, dtype=np.float64)

        # Compute counts
        for idx, cls in enumerate(classes):
            class_mask = dataset.y == cls
            class_counts[idx] = np.sum(class_mask)
            feature_counts[idx, :] = np.sum(dataset.X[class_mask], axis=0)

        # Compute class prior probabilities
        self.class_prior = (class_counts + self.smoothing) / (n_samples + n_classes * self.smoothing)

        # Apply Laplace smoothing and compute feature probabilities
        smoothed_feature_counts = feature_counts + self.smoothing
        smoothed_class_counts = class_counts + 2 * self.smoothing
        self.feature_probs = smoothed_feature_counts / smoothed_class_counts[:, np.newaxis]

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the class labels for a given set of samples.

        Parameters
        ----------
        dataset: Dataset
            The test dataset.

        Returns
        -------
        predictions: np.ndarray
            An array of predicted class labels for the test dataset.
        """
        predictions = []
        for sample in dataset.X:
            class_probs = []
            for c in range(len(self.class_prior)):
                # Compute probability for each class
                prob = np.prod(sample * self.feature_probs[c] + (1 - sample) * (1 - self.feature_probs[c]))
                prob *= self.class_prior[c]
                class_probs.append(prob)
            # Choose the class with the highest probability
            predictions.append(np.argmax(class_probs))
        return np.array(predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Compute the accuracy of the model on the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The test dataset.
        predictions: np.ndarray
            An array of predicted class labels.

        Returns
        -------
        accuracy: float
            The accuracy of the model.
        """

        return accuracy(dataset.y, predictions)