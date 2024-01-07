from pathlib import Path
from typing import Union

import numpy as np
import torch
from safetensors.torch import save_model
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import MusicDataset
from .logger import Logger
from .model import LinearModel


class Trainer:
    def __init__(
        self,
        model: LinearModel,
        optimizer: Optimizer,
        train_dataset: MusicDataset,
        valid_dataset: MusicDataset,
        logger: Logger,
        epochs: int = 5,
        batch_size: int = 1024,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False
        )
        self.logger = logger
        self.loss = nn.MSELoss()

        self.epochs = epochs

        self.mean_x = train_dataset.mean_x
        self.std_x = train_dataset.std_x
        self.mean_y = train_dataset.mean_y
        self.std_y = train_dataset.std_y

    def train(self, save_name: Path = None):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}")
            self.model.train()
            for batch in tqdm(self.train_dataloader, desc="Training"):
                x = batch["x"]
                y = batch["y"]
                y_predicted = self.model(x).squeeze()
                loss_batch = self.loss(y_predicted, y)
                loss_batch.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                accuracy = torch.sum(
                    y.to(torch.long) == torch.round(y_predicted).to(torch.long)
                ) / len(y)
                self.logger.update(
                    {
                        "train_loss_batch": loss_batch.item(),
                        "train_accuracy_batch": float(accuracy),
                    }
                )

            self.model.eval()
            with torch.no_grad():
                loss_total_valid = 0
                total_len = 0
                y_cls = []
                y_predicted_cls = []
                for batch in tqdm(self.valid_dataloader, desc="Validation"):
                    x = batch["x"]
                    y = batch["y"]
                    y_predicted = self.model(x).squeeze()
                    loss_total_valid += self.loss(y_predicted, y).item() * len(y)
                    total_len += len(y)
                    y_cls.extend(y.to(torch.long))
                    y_predicted_cls.extend(torch.round(y_predicted).to(torch.long))
                self.logger.update(
                    {
                        "validation_loss": loss_total_valid / total_len,
                        "validation_accuracy": float(
                            np.mean(np.array(y_cls) == np.array(y_predicted_cls))
                        ),
                    }
                )
            self.logger.finalize()
        if save_name:
            self.save(save_name)

    def save(self, save_name: Union[Path, str] = "./models/model.safetensors"):
        if save_name is str:
            save_name = Path(save_name)
        save_name.mkdir(parents=True, exist_ok=True)
        save_model(self.model.actual_model, save_name)
        with open(save_name.with_name("preprocessing.json"), "w") as f:
            f.write(str(self.mean_x) + "\n" + f.write(str(self.std_x)) + "\n")
            f.write(str(self.mean_y) + "\n" + f.write(str(self.std_y)))
