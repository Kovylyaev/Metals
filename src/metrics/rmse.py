import numpy as np
from torchmetrics import Metric


class RMSE(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, preds, y):
        if preds.shape[0] != y.shape[0]:
            raise ValueError("predictions and true answers must have the same shape")
        self.squared_error = ((preds - y)**2).mean()

    def compute(self):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8
        """
        return np.sqrt(self.squared_error)
    