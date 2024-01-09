from pathlib import Path

import numpy as np
import torch
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
        save_name: Path = None,
    ):
        thread_count = torch.get_num_threads()
        torch.set_num_threads(thread_count)
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

        self.save_name = save_name

    def train(self):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}")
            self.model.train()
            for batch in tqdm(self.train_dataloader, desc="Training"):
                features = batch["features"]
                target = batch["target"]
                predicted = self.model(features).squeeze()
                loss_batch = self.loss(predicted, target)
                loss_batch.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                accuracy = torch.sum(
                    target.to(torch.long) == torch.round(predicted).to(torch.long)
                ) / len(target)
                self.logger.update(
                    {
                        "train_loss_batch": loss_batch.item(),
                        "train_accuracy_batch": float(accuracy),
                    },
                    on="train",
                )

            self.model.eval()
            with torch.no_grad():
                loss_total_valid = 0
                total_len = 0
                target_cls = []
                predicted_cls = []
                for batch in tqdm(self.valid_dataloader, desc="Validation"):
                    features = batch["features"]
                    target = batch["target"]
                    predicted = self.model(features).squeeze()
                    loss_total_valid += self.loss(predicted, target).item() * len(target)
                    total_len += len(target)
                    target_cls.extend(target.to(torch.long))
                    predicted_cls.extend(torch.round(predicted).to(torch.long))
                self.logger.update(
                    {
                        "validation_loss": loss_total_valid / total_len,
                        "validation_accuracy": float(
                            np.mean(np.array(target_cls) == np.array(predicted_cls))
                        ),
                    },
                    on="val",
                )
        if self.save_name:
            self.model.save(self.save_name)
        self.logger.finalize()
