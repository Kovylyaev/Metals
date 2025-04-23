import torch
from torchmetrics import Metric


class Accuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, preds, y):
        if preds.shape[0] != y.shape[0]:
            raise ValueError("predictions and true answers must have the same shape")
        
        self.predicted_classes = torch.argmax(preds, axis=1)
        self.y = y
        self.batch_size = len(y)

    def compute(self):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8
        """
        return (torch.sum(self.predicted_classes == self.y) / self.batch_size).item()
    