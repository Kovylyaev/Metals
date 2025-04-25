import torch


class ResNet34Model():
    def __init__(self, weights, output_dim):
        self.model = torch.hub.load("pytorch/vision", "resnet34", weights=weights)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=output_dim, bias=True)