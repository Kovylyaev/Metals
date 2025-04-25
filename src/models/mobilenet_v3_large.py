import torch


class mobilenet_v3_largeModel():
    def __init__(self, weights, output_dim):
        self.model = torch.hub.load("pytorch/vision", "mobilenet_v3_large", weights=weights)
        self.model.classifier[3] = torch.nn.Linear(in_features=1280, out_features=1, bias=True)