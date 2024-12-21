from unittest import TestCase

import numpy as np


from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv

from si.models.knn_regressor import KNNRegressor

from si.model_selection.split import train_test_split

class TestKNNRegressor(TestCase):
    def setUp(self):
        """
        Configura o conjunto de dados de CPU para os testes.
        """
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        """
        Testa o método fit do KNNRegressor.
        """
        knn = KNNRegressor(k=3)

        knn.fit(self.dataset)

        self.assertTrue(np.all(self.dataset.features == knn.dataset.features))
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y))

    def test_predict(self):
        """
        Testa o método predict do KNNRegressor.
        """
        knn = KNNRegressor(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.2, random_state=42)
        knn._fit(train_dataset)
        predictions = knn._predict(test_dataset)

        self.assertEqual(predictions.shape[0], test_dataset.X.shape[0])

        self.assertTrue(np.all(predictions >= np.min(train_dataset.y)))
        self.assertTrue(np.all(predictions <= np.max(train_dataset.y)))

    def test_score(self):
        """
        Testa o método score do KNNRegressor (RMSE).
        """
        knn = KNNRegressor(k=3)

        train_dataset, test_dataset = train_test_split(self.dataset)

        knn.fit(train_dataset)
        score = knn.score(test_dataset)

        # RMSE deve ser um número positivo
        self.assertGreaterEqual(score, 0)