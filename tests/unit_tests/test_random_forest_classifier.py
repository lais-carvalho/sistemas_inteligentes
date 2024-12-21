from unittest import TestCase

from datasets import DATASETS_PATH

import os


from si.model_selection.split import train_test_split
from si.models.random_forest_classifier import RandomForestClassifier
from src.si.io.csv_file import read_csv
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class TestRandomForestClassifier(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

        self.rf = RandomForestClassifier(n_estimators=10, max_depth=5, mode="gini", seed=42)

    def test_fit(self):

        self.rf.fit(self.train_dataset)

        self.assertEqual(len(self.rf.trees), 10, f"Esperado 10 árvores, mas o modelo tem {len(self.rf.trees)} árvores.")

    def test_predict(self):

        self.rf.fit(self.train_dataset)

        # Fazer previsões
        predictions = self.rf.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0])

    def test_score(self):
        self.rf.fit(self.train_dataset)

        predictions = self.rf.predict(self.test_dataset)

        accuracy_ = accuracy(self.test_dataset.y, predictions)

        self.assertAlmostEqual(accuracy_, 1.0, places=2)
