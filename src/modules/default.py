import lightning as L
import torch
import torchmetrics
from hydra.utils import instantiate

from src.metrics import RMSE

class DefaultModule(L.LightningModule):
    def __init__(self, cfg, scaler):
        """
            Инициализирует модель.
            
            Args:
                model_config - конфигурация модели
                training_config - конфигурация обучения
        """
        super().__init__()
        self.scaler = scaler

        self.model = instantiate(cfg.model).model
        self.optimizer = instantiate(cfg.optimizer, params=self.model.parameters())
        self.criterion = instantiate(cfg.criterion)
        self.scheduler = instantiate(cfg.scheduler, optimizer=self.optimizer) if cfg.scheduler['_target_'] else None

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "rmse": RMSE(),
            },
            prefix="train_"
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
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
        x, y, weights = batch
        y = torch.tensor(self.scaler.transform(y.cpu().reshape(-1, 1))).float().to(torch.device('mps'))
        preds = self.model(x)
        loss = self.criterion(preds, y, weights)

        batch_metrics = self.train_metrics(self.scaler.inverse_transform(preds.detach().cpu()), self.scaler.inverse_transform(y.cpu()))
        self.log_dict(batch_metrics, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
            Шаг валидации
            
            Args:
                batch - tuple (x, y)
                batch_idx - индекс батча
        """
        x, y, _ = batch
        y = torch.tensor(self.scaler.transform(y.cpu().reshape(-1, 1))).float().to(torch.device('mps'))
        preds = self.model(x)
        loss = self.criterion(preds, y, None)

        batch_metrics = self.val_metrics(self.scaler.inverse_transform(preds.cpu()), self.scaler.inverse_transform(y.cpu()))
        self.log_dict(batch_metrics, on_step=False, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """
            Шаг тестирования
            
            Args:
                batch - tuple (x, y)
                batch_idx - индекс батча
        """
        x, y, _ = batch
        y = torch.tensor(self.scaler.transform(y.cpu().reshape(-1, 1))).float().to(torch.device('mps'))
        preds = self.model(x)
        loss = self.criterion(preds, y, None)

        batch_metrics = self.test_metrics(self.scaler.inverse_transform(preds.cpu()), self.scaler.inverse_transform(y.cpu()))
        self.log_dict(batch_metrics, on_step=False, on_epoch=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss, batch_metrics
    
    def predict_step(self, batch, batch_idx):
        x, y, _ = batch
        return self.scaler.inverse_transform(self.model(x).cpu()), y

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
