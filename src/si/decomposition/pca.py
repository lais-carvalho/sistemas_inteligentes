import numpy as np

from src.si.base.transformer import Transformer
from src.si.data.dataset import Dataset

class PCA(Transformer):
    """
    parameters:
        n_components – number of components
    estimated parameters:
        mean – mean of the samples
        components – the principal components (a matrix where each row is an eigenvector corresponding to a principal component)
        explained_variance – the amount of variance explained by each principal component (a vector of eigenvalues)
    methods:
        fit – estimates the mean, principal components, and explained variance
        transform – calculates the reduced dataset using the principal components
    """

    def __init__(self, n_components: int, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, dataset: Dataset) -> 'PCA':
        self.mean = np.mean(dataset.X, axis=0)
        X_centered = dataset.X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]

        return self

    def _transform(self, dataset: Dataset) -> Dataset:

        X_centered = dataset.X - self.mean
        X_reduced = np.dot(X_centered, self.components)

        features_reduced = [f'PC{i + 1}' for i in range(self.n_components)]
        return Dataset(X=X_reduced, y=dataset.y, features=features_reduced, label=dataset.label)