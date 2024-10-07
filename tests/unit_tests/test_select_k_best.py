import os
from unittest import TestCase

from datasets import DATASETS_PATH

from si.io.csv_file import read_csv


class TestSelectKBest(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        select_k_best = SelectKBest(score_func = f_classification, k = 2)

        select_k_best.fit(self.dataset)
        self.assertTrue(select_k_best.F != None)
        self.assertTrue(select_k_best.p != None)

    def test_transform(self):
        select_k_best = SelectKBest(socroe_func = f_classification, k = 2)

        select_k_best.fit(self, dataset)
        new_dataset = select_k_best.transform(self.dataset)

        self.assertTrue(len(new_dataset.features), len(self.dataset.features))
        self.assertless(new_dataset.X.shape[1], self.dataset.X.shape[1])
