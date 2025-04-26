import torch


class efficientnet_v2_mModel():
    def __init__(self, weights, output_dim):
        self.model = torch.hub.load("pytorch/vision", "efficientnet_v2_m", weights=weights)
        self.model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=output_dim, bias=True)