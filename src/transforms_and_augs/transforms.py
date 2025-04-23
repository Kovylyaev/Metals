from torchvision.transforms import v2
import torch


def transform(pic):
    trans = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomCrop(size=(224, 224)),
    ])
    return trans(pic)