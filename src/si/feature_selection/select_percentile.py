import numpy as np
from typing import Callable
from src.si.base.transformer import Transformer
from src.si.data.dataset import Dataset
from src.si.statistics.f_classification import f_classification

class SelectPercentile(Transformer):
    """
    Select features according to the percentile.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.

    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    percentile: int, default=10
        Percentile to select.

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """

    def __init__(self, score_func: Callable = f_classification, percentile: int = 10, **kwargs):
        """
        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        percentile: int, default=10
        """
        super().__init__(**kwargs)
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        It fits SelectPercentile to compute the F scores and p-values.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting the features by percentile.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the top features based on the specified percentile.
        """
        threshold = np.percentile(self.F, 100 - self.percentile)
        idxs = np.where(self.F >= threshold)[0]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)