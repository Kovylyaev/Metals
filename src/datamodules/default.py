import lightning as L
from pathlib import Path
from torch.utils.data import random_split, DataLoader
import subprocess
from hydra.utils import instantiate

from src.transforms_and_augs import transform, augment


class DefaultDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.img_dir = Path.cwd().parent.joinpath(cfg.training.img_dir)
        self.answer_file = Path.cwd().parent.joinpath(cfg.training.answer_file)
        self.batch_size = cfg.training.batch_size
        self.cfg = cfg
        self.transform = transform
        self.augment = augment

    def prepare_data(self):
        try:
            print("Подготовка данных...")
            subprocess.run(['make', '-C', str(Path.cwd().parent), 'prepare_data'], check=True, text=True, capture_output=True)
            print("Подготовка данных завершена")
        except subprocess.CalledProcessError as e:
            print("Ошибка при подготовке данных:")
            print(e.stderr)
            raise e
        pass
            
    def setup(self, stage: str):
        # Assign train/test datasets for use in dataloaders
        train_imgs, val_imgs, test_imgs = random_split(list(self.img_dir.iterdir()), [self.cfg.training.train_size, self.cfg.training.val_size, self.cfg.training.test_size])

        self.train_dataset = instantiate(self.cfg.dataset,
            img_paths=train_imgs,
            answers_file=str(self.answer_file),
            transform=self.transform,
            augment=self.augment,
            train=True
        )
        self.val_dataset = instantiate(self.cfg.dataset,
            img_paths=val_imgs,
            answers_file=str(self.answer_file),
            transform=self.transform,
            augment=self.augment,
            train=False
        )
        self.test_dataset = instantiate(self.cfg.dataset,
            img_paths=test_imgs,
            answers_file=str(self.answer_file),
            transform=self.transform,
            augment=self.augment,
            train=False
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
