import numpy as np

from src.si.base.model import Model
from src.si.data.dataset import Dataset
from src.si.metrics.accuracy import accuracy

class StackingClassifier(Model):
    def __init__(self, models, final_model, **kwargs):
        super().__init__(**kwargs)
        self.models = models
        self.final_model = final_model

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':

        base_predictions = []
        for model in self.models:
            model.fit(dataset)
            predictions = model.predict(dataset)
            base_predictions.append(predictions)

        self.base_predictions = np.column_stack(base_predictions)

        self.final_model.fit(Dataset(self.base_predictions, dataset.y))
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:

        base_predictions = []
        for model in self.models:
            predictions = model.predict(dataset)
            base_predictions.append(predictions)

        stacked_predictions = np.column_stack(base_predictions)

        return self.final_model.predict(Dataset(stacked_predictions, None))

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:

        return accuracy(dataset.y, predictions)
