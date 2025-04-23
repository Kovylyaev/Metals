import lightning as L
import torch
import torchmetrics
from hydra.utils import instantiate

from src.metrics import Accuracy

class DefaultModule(L.LightningModule):
    def __init__(self, cfg):
        """
            Инициализирует модель.
            
            Args:
                model_config - конфигурация модели
                training_config - конфигурация обучения
        """
        super().__init__()

        self.model = instantiate(cfg.model).model
        self.optimizer = instantiate(cfg.optimizer, params=self.model.parameters())
        self.criterion = instantiate(cfg.criterion)
        self.scheduler = instantiate(cfg.scheduler) if cfg.scheduler['_target_'] else None

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "acc": Accuracy(),
            },
            prefix="train_"
        )
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, x):
        """
            Проход вперёд для инференса

            Args:
                x - входные данные
        """
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        """
            Шаг обучения
            
            Args:
                batch - tuple (x, y)
                batch_idx - индекс батча
        """
        x, y = batch
        preds = self.model(x)
        loss = self.criterion(preds, y)

        batch_metrics = self.train_metrics(preds, y)
        self.log_dict(batch_metrics, on_step=True, on_epoch=True)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
            Шаг валидации
            
            Args:
                batch - tuple (x, y)
                batch_idx - индекс батча
        """
        x, y = batch
        preds = self.model(x)
        loss = self.criterion(preds, y)

        batch_metrics = self.test_metrics(preds, y)
        self.log_dict(batch_metrics, on_step=True, on_epoch=True)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        """
            Шаг тестирования
            
            Args:
                batch - tuple (x, y)
                batch_idx - индекс батча
        """
        x, y = batch
        preds = self.model(x)
        loss = self.criterion(preds, y)

        batch_metrics = self.test_metrics(preds, y)
        self.log_dict(batch_metrics, on_step=True, on_epoch=True)
        self.log("test_loss", loss)
        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.test_metrics.reset()
    
    def on_test_epoch_end(self):
        self.test_metrics.reset()
    
    def configure_optimizers(self):
        """
        Возвращает оптимизатор и, если задан, планировщик скорости обучения.
        """
        if self.scheduler is None:
            return self.optimizer
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
        }
