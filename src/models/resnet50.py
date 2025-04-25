import torch


class ResNet50Model():
    def __init__(self, weights, output_dim):
        self.model = torch.hub.load("pytorch/vision", "resnet50", weights=weights)
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=output_dim, bias=True)