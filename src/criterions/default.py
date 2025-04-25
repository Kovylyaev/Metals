import torch

class WeightedMSELoss:
    def __call__(self, preds, targets, weights):
        if weights is None:
            return torch.mean((preds - targets)**2) * 5
        return torch.mean((preds - targets)**2 * weights) * 5