import numpy as np

from si.base.transformer import Transformer
import numpy

class SelectKBest(Transformer):

    def __init__(self, score_func, k, **kwargs):
        super().__init__(**kwargs)
        slef.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def _fit(self, dataset):
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset):
        idx = np.argsort(self.F)
        mask = idx[-self.k:]
        new_x = dataset.X[:, mask]
        new_features = dataset.features[mask]

        return Dataset(X = new_x, features = new_features, y = dataset.y, label = dataset.label)
