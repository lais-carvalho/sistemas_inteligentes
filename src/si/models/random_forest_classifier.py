import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier(Model):
    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2, max_depth=None, mode="gini", seed=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []  # trained trees

    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':

        if self.seed is not None:
            np.random.seed(self.seed)

        if self.max_features is None:
            self.max_features = int(np.sqrt(dataset.X.shape[1]))

        for _ in range(self.n_estimators):

            n_samples = dataset.X.shape[0]
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_X = dataset.X[bootstrap_indices]
            bootstrap_y = dataset.y[bootstrap_indices]

            feature_indices = np.random.choice(dataset.X.shape[1], size=self.max_features, replace=False)
            bootstrap_X_features = bootstrap_X[:, feature_indices]

            tree = DecisionTreeClassifier(max_depth=self.max_depth, mode=self.mode)
            tree.fit(Dataset(bootstrap_X_features, bootstrap_y))  # Train on bootstrap data

            self.trees.append((feature_indices, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        predictions = []

        # Faz previsões para cada árvore na floresta
        for feature_indices, tree in self.trees:
            # Seleciona as características relevantes para o dataset de teste
            X_test_features = dataset.X[:, feature_indices]
            predictions.append(tree.predict(Dataset(X_test_features, dataset.y)))

        # Converte as previsões para um array numpy e transpõe para votação majoritária
        predictions = np.array(predictions).T

        # Calcula a votação majoritária por linha
        majority_votes = []
        for row in predictions:
            values, counts = np.unique(row, return_counts=True)
            majority_votes.append(values[np.argmax(counts)])

        return np.array(majority_votes)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:

        return accuracy(dataset.y, predictions)