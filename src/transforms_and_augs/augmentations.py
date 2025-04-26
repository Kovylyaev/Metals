from torchvision.transforms import v2


def augment(pic):
    augs = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=(-180, 180)),
        v2.GaussianBlur(kernel_size=5, sigma=(0.0001, 1)),
    ])
    return augs(pic)