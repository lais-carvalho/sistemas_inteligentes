from unittest import TestCase

from datasets import DATASETS_PATH

import numpy as np

import os
from src.si.io.csv_file import read_csv

from src.si.model_selection.split import train_test_split
from src.si.model_selection.split import stratified_train_test_split

class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

        _, self.dataset.y = np.unique(self.dataset.y, return_inverse=True)

    def test_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)


    def test_stratified_train_test_split(self):

        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)

        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)

        original_class_proportions = np.bincount(self.dataset.y) / self.dataset.shape()[0]

        test_class_proportions = np.bincount(test.y) / test.shape()[0]

        train_class_proportions = np.bincount(train.y) / train.shape()[0]

        for i in range(len(original_class_proportions)):
            self.assertEqual(original_class_proportions[i], test_class_proportions[i])
            self.assertEqual(original_class_proportions[i], train_class_proportions[i])