import torch


class vit_l_16Model():
    def __init__(self, weights, output_dim):
        self.model = torch.hub.load("pytorch/vision", "vit_l_16", weights=weights)
        self.model.heads[0] = torch.nn.Linear(in_features=1024, out_features=1, bias=True)