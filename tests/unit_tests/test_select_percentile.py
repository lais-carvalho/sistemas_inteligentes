from unittest import TestCase
from datasets import DATASETS_PATH

import os
from src.si.feature_selection.select_percentile import SelectPercentile
from src.si.io.csv_file import read_csv

from src.si.statistics.f_classification import f_classification


class TestSelectPercentile(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        """
        Testa se o método `fit` calcula corretamente os escores e valores-p.
        """
        select_percentile = SelectPercentile(score_func=f_classification, percentile=50)

        select_percentile.fit(self.dataset)

        # Verifica se os escores e valores-p foram calculados, sendo calculado terá resultados
        self.assertTrue(select_percentile.F.shape[0] > 0)
        self.assertTrue(select_percentile.p.shape[0] > 0)

    def test_transform(self):
        """
        Testa se o método `transform` seleciona corretamente as features com base no percentil.
        """
        select_percentile = SelectPercentile(score_func=f_classification, percentile=50)

        select_percentile.fit(self.dataset)

        new_dataset = select_percentile.transform(self.dataset)

        # Calcula o número esperado de features baseado no percentil
        expected_num_features = int(len(self.dataset.features) * 0.5)

        # Verifica se o novo dataset tem o número esperado de features
        self.assertEqual(len(new_dataset.features), expected_num_features)
        self.assertEqual(new_dataset.X.shape[1], expected_num_features) #o número de colunas precisa ser igual ao número de features selecionados
        self.assertLess(len(new_dataset.features), len(self.dataset.features)) #como foi selecionado os melhores features, o novo dataset precisa ser menor