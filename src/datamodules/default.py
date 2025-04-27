import lightning as L
from pathlib import Path
from torch.utils.data import random_split, DataLoader
import subprocess
from hydra.utils import instantiate
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

        self._prepare_data()

    def _prepare_data(self):
        # try:
        #     print("Подготовка данных...")
        #     subprocess.run(['make', '-C', str(Path.cwd().parent), 'prepare_data', f'target_column=Fe'], check=True, text=True, capture_output=True)
        #     print("Подготовка данных завершена")
        # except subprocess.CalledProcessError as e:
        #     print("Ошибка при подготовке данных:")
        #     print(e.stderr)
        #     raise e
        self.train_imgs, self.val_imgs, self.test_imgs = random_split(list(self.img_dir.iterdir()), [self.cfg.training.train_size, self.cfg.training.val_size, self.cfg.training.test_size])
        self.train_imgs = list(map(str, list(self.train_imgs)))
        self.val_imgs = list(map(str, list(self.val_imgs)))
        self.test_imgs = list(map(str, list(self.test_imgs)))
        
        answer_file = pd.read_csv(self.answer_file, index_col=0)
        self.scaler = MinMaxScaler()
        self.scaler.fit(answer_file.loc[self.train_imgs, self.cfg.target].values.reshape(-1, 1))
            
    def setup(self, stage: str):
        # Assign train/test datasets for use in dataloaders

        self.train_dataset = instantiate(self.cfg.dataset,
            img_paths=self.train_imgs,
            answers_file=str(self.answer_file),
            target_column=self.cfg.target,
            transform=self.transform,
            augment=self.augment,
            train=True
        )
        self.val_dataset = instantiate(self.cfg.dataset,
            img_paths=self.val_imgs,
            answers_file=str(self.answer_file),
            target_column=self.cfg.target,
            transform=self.transform,
            augment=self.augment,
            train=False
        )
        self.test_dataset = instantiate(self.cfg.dataset,
            img_paths=self.test_imgs,
            answers_file=str(self.answer_file),
            target_column=self.cfg.target,
            transform=self.transform,
            augment=self.augment,
            train=False
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)#, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)#, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)#, num_workers=4)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)#, num_workers=4)
