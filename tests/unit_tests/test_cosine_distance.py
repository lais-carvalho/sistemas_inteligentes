from unittest import TestCase

import numpy as np
from datasets import DATASETS_PATH

import os
from src.si.io.csv_file import read_csv

from src.si.statistics.cosine_distance import cosine_distance

class TestCosineDistance(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_cosine_distance(self):
        x = np.array([1, 2, 3])
        y = np.array([[1, 2, 3], [4, 5, 6]])
        our_distance = cosine_distance(x, y)

        from sklearn.metrics.pairwise import cosine_distances
        sklearn_distance = cosine_distances(x.reshape(1, -1), y)
        assert np.allclose(our_distance, sklearn_distance)