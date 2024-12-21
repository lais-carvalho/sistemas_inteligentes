from unittest import TestCase

from datasets import DATASETS_PATH

import os

from si.io.data_file import read_data_file
from si.metrics.accuracy import accuracy
from si.model_selection.randomized_search import randomized_search_cv
from si.models.logistic_regression import LogisticRegression

import numpy as np
class TestRandomizedSearchCV(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        # Carrega o dataset
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_randomized_search_k_fold_cross_validation(self):

        model = LogisticRegression()

        # Definir o grid de hiperparâmetros
        parameter_grid_ = {
            'l2_penalty': (1, 10),
            'alpha': (0.001, 0.0001),
            'max_iter': (1000, 2000)
        }

        # Número de iterações para o random search
        n_iter = 5

        # Executa a busca aleatória com validação cruzada
        results_ = randomized_search_cv(
            model,
            self.dataset,
            hyperparameter_grid=parameter_grid_,
            cv=3,
            n_iter=n_iter
        )

        # Verifica os resultados
        self.assertEqual(len(results_["scores"]), n_iter)

        # Verifica as melhores combinações de hiperparâmetros
        best_hyperparameters = results_['best_hyperparameters']
        self.assertEqual(len(best_hyperparameters), 3)

        # Verifica o melhor score
        best_score = results_['best_score']
        self.assertEqual(np.round(best_score, 2), 0.97)