import torch


class mobilenet_v3_smallModel():
    def __init__(self, weights, output_dim):
        self.model = torch.hub.load("pytorch/vision", "mobilenet_v3_small", weights=weights)
        self.model.classifier[3] = torch.nn.Linear(in_features=1024, out_features=output_dim, bias=True)