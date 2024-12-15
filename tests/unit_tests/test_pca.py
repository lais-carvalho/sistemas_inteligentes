from unittest import TestCase
from datasets import DATASETS_PATH

import os
from src.si.decomposition.pca import PCA
from src.si.io.csv_file import read_csv

class TestPCA(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        pca = PCA(n_components=3)

        pca.fit(self.dataset)


        self.assertTrue(pca.components is not None)
        self.assertEqual(pca.components.shape[1], 3)

        # Verifica se a variância explicada foi calculada
        self.assertTrue(pca.explained_variance is not None)
        self.assertEqual(len(pca.explained_variance), 3)

    def test_transform(self):
        """
        Testa se o método `transform` transforma corretamente o dataset nas componentes principais.
        """
        pca = PCA(n_components=1)

        pca.fit(self.dataset)

        new_dataset = pca.transform(self.dataset)

        self.assertEqual(len(new_dataset.features), 1)
        self.assertEqual(new_dataset.X.shape[1], 1)