import torch


class DefaultModel():
    def __init__(self, weights, num_classes):
        self.model = torch.hub.load("pytorch/vision", "resnet18", weights=weights)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)