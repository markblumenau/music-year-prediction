import os
import warnings

import dvc.api
import torch
from torch import nn


class LinearBlock((nn.Module)):
    def __init__(self, hidden_size: int = 100, input_size: int = None):
        super(LinearBlock, self).__init__()
        if input_size:
            self.fc = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
            )

    def forward(self, x):
        return self.fc(x)


class LinearModel((nn.Module)):
    def __init__(
        self,
        block_count: int = 3,
        input_size: int = 100,
        hidden_size: int = 1000,
        output_size: int = 1,
        mean_x: float = 0.0,
        std_x: float = 0.0,
        mean_y: float = 0.0,
        std_y: float = 0.0,
    ):
        super(LinearModel, self).__init__()
        input = LinearBlock(hidden_size, input_size)
        hidden_blocks = []
        for _i in range(block_count):
            hidden_blocks.append(LinearBlock(hidden_size))
        output = nn.Linear(hidden_size, output_size)
        self.actual_model = nn.Sequential(input, *hidden_blocks, output)
        self.mean_x = mean_x
        self.std_x = std_x
        self.mean_y = mean_y
        self.std_y = std_y

    def forward(self, x):
        # I am fully aware that the following lines are rather bad
        # But the whole point of this repo is to learn MLOps, not to make great models
        x = (x - self.mean_x) / self.std_x
        return self.actual_model(x) * self.std_y + self.mean_y

    def load(self, pull_dvc: bool = True, load_name: str = None):
        if pull_dvc:
            with dvc.api.open("models/model.pth") as f:
                self.actual_model.load_state_dict(torch.load(f))
                self.actual_model.eval()
            with dvc.api.open("models/model.pthinfo") as file:
                with open(file, "r") as f:
                    self.mean_x = float(f.readline())
                    self.std_x = float(f.readline())
                    self.mean_y = float(f.readline())
                    self.std_y = float(f.readline())
        elif load_name:
            if os.path.isfile(load_name):
                self.actual_model.load_state_dict(torch.load(load_name))
                self.actual_model.eval()
            else:
                warnings.warn(
                    "You tried to load model with a nonexistent weight file, skipping.",
                    stacklevel=2,
                )
            if os.path.isfile(load_name + "info"):
                with open(load_name + "info", "r") as f:
                    self.mean_x = float(f.readline())
                    self.std_x = float(f.readline())
                    self.mean_y = float(f.readline())
                    self.std_y = float(f.readline())
            else:
                warnings.warn(
                    "You tried to load model with a nonexistent train info file, predictions are probably useless.",
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "You tried to load model without a weight file, skipping.", stacklevel=2
            )
