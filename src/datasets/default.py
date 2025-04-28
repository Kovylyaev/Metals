import pandas as pd
import torch
from PIL import Image


class DefaultDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths: list[str], answers_file: str, target_column: str, transform, augment, train: bool):
        """Initializes Dataset with passed files.
        Args:
            img_paths: paths (Path) to files with microstructures,
            answers: path to file with dict: path (str) to C concentration,
            transform: transform to apply to images,
            augment: augment to apply to images,
            train: (bool) is it dataset for training or not
        """
        self.img_paths = img_paths
        self.answers = pd.read_csv(answers_file, index_col=0)
        self.target_column = target_column
        self.transform = transform
        self.augment = augment
        self.train = train

        float64_cols = list(self.answers.select_dtypes(include='float64'))
        self.answers[float64_cols] = self.answers[float64_cols].astype('float32')


    def __getitem__(self, idx: int):
        """Returns the object by given index.
        Args:
            idx - index of the image.
        Returns:
            Image with microstructure,
            target concentration,
            sample weight
            scaler
        """

        path = self.img_paths[idx]

        img = self.transform(Image.open(path))
        answer = self.answers.loc[path][self.target_column]
        weight = self.answers.loc[path]['weight']

        if self.train:
            img = self.augment(img)

        return img, answer, weight


    def __len__(self):
        """Returns num of images."""

        return len(self.img_paths)