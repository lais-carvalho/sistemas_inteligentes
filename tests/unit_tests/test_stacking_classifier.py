from unittest import TestCase

from datasets import DATASETS_PATH

import os

from si.ensemble.stacking_classifier import StackingClassifier
from si.model_selection.split import stratified_train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from src.si.io.data_file import read_data_file


class TestStackingClassifier(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = stratified_train_test_split(self.dataset, test_size=0.25)

        decision_tree = DecisionTreeClassifier()
        knn = KNNClassifier()
        logistic_regression = LogisticRegression()

        self.final_model = LogisticRegression()

        self.stack = StackingClassifier(models=[decision_tree, knn, logistic_regression], final_model=self.final_model)

    def test_fit(self):
        # Treinar o modelo
        self.stack.fit(self.train_dataset)

        # Verificar se o modelo final foi treinado (isso pode ser feito verificando se a base de predições foi gerada)
        self.assertTrue(hasattr(self.stack, 'base_predictions'),
                        "O modelo final não foi treinado corretamente com as predições.")

    def test_predict(self):
        # Treinar o modelo
        self.stack.fit(self.train_dataset)

        # Fazer previsões
        predictions = self.stack.predict(self.test_dataset)

        # Verificar se o número de previsões é igual ao número de amostras no conjunto de teste
        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0],
                         f"O número de previsões ({predictions.shape[0]}) não é igual ao número de amostras no conjunto de teste ({self.test_dataset.X.shape[0]})")

    def test_score(self):

        self.stack.fit(self.train_dataset)

        predictions = self.stack.predict(self.test_dataset)

        accuracy_ = self.stack.score(self.test_dataset)

        self.assertAlmostEqual(round(accuracy_, 2), 0.95, delta=0.05,
                               msg=f"A acurácia esperada é 0.95, mas obteve {round(accuracy_, 2)}") #diferença de até 0.05