import json
import os
import shutil
import warnings
from pathlib import Path

import dvc.api
import pandas as pd
import torch
from safetensors.torch import load_model, save_model
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

    def forward(self, input):
        return self.fc(input)


class LinearModel((nn.Module)):
    def __init__(
        self,
        block_count: int = 3,
        input_size: int = 90,
        hidden_size: int = 1000,
        output_size: int = 1,
        mean_features: float = 0.0,
        std_features: float = 0.0,
        mean_target: float = 0.0,
        std_target: float = 0.0,
    ):
        super(LinearModel, self).__init__()
        self.input_size = input_size
        input = LinearBlock(hidden_size, input_size)
        hidden_blocks = []
        for _i in range(block_count):
            hidden_blocks.append(LinearBlock(hidden_size))
        output = nn.Linear(hidden_size, output_size)
        self.actual_model = nn.Sequential(input, *hidden_blocks, output)
        self.preprocesser = {
            "mean_features": mean_features,
            "mean_target": mean_target,
            "std_features": std_features,
            "std_target": std_target,
        }

    def forward(self, input):
        # I am fully aware that the following lines are rather bad
        # But the whole point of this repo is to learn MLOps, not to make great models
        output = (input - self.preprocesser["mean_features"]) / self.preprocesser[
            "std_features"
        ]
        return (
            self.actual_model(output) * self.preprocesser["std_target"]
            + self.preprocesser["mean_target"]
        )

    def predict(self, input: Path = None, demo: bool = True):
        df = None
        if input is not None:
            df = pd.read_csv(input, header=None)
        elif demo:
            with dvc.api.open("./data/test_sample.txt") as file:
                df = pd.read_csv(file, header=None)
        if df is None:
            raise Exception("Nothing to predict on!")
        if df.shape[1] != self.input_size:
            raise ValueError(
                "Your items have a different size from the ones used for training."
            )
        with torch.no_grad():
            return self.forward(torch.tensor(df.values).float()).numpy().reshape(-1)

    def save(self, save_name: Path):
        save_name.parents[0].mkdir(parents=True, exist_ok=True)
        save_model(self.actual_model, save_name)
        with open(save_name.with_name("preprocessing.json"), "w") as file:
            json.dump(self.preprocesser, file)

    def load(self, pull_dvc: bool = True, load_name: Path = None):
        if pull_dvc:
            # Because dvc.open returns a file
            # and safetensors wants a path
            # and under the hood shutil errors on samefile
            fs = dvc.api.DVCFileSystem()
            try:
                fs.get_file("models/model.safetensors", "models/model.safetensors")
            except shutil.SameFileError:
                pass
            try:
                fs.get_file("models/preprocessing.json", "models/preprocessing.json")
            except shutil.SameFileError:
                pass
            load_model(self.actual_model, "./models/model.safetensors")
            with open("./models/preprocessing.json", "r") as file:
                self.preprocesser = json.load(file)

        elif load_name:
            if os.path.isfile(load_name):
                load_model(self.actual_model, load_name)
            else:
                warnings.warn(
                    "You tried to load model with a nonexistent weight file, skipping.",
                    stacklevel=2,
                )
            if os.path.isfile(load_name.with_name("preprocessing.json")):
                with open(load_name.with_name("preprocessing.json"), "r") as file:
                    self.preprocesser = json.load(file)
            else:
                warnings.warn(
                    "You tried to load model with a nonexistent train info file, predictions are probably useless.",
                    stacklevel=2,
                )

        else:
            warnings.warn(
                "You tried to load model without a weight file, skipping.", stacklevel=2
            )
        self.actual_model.eval()
