import dvc.api
import pandas as pd
from omegaconf import DictConfig
from torch import optim

from .dataset import MusicDataset


def get_optimizer(params, cfg: DictConfig):
    if cfg.optim.get("optim_type", "Adam") == "Adam":
        return optim.Adam(params, lr=cfg.optim.get("lr", 1e-3))
    else:
        return optim.SGD(params, lr=cfg.optim.get("lr", 1e-2))


def make_datasets(train_size: int = 463715):
    with dvc.api.open("./data/YearPredictionMSD.txt") as f:
        df = pd.read_csv(f, header=None)
        # Yep, one letter names, but they are very obvious
        # y = f(x)
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values
        x_train = X[:train_size, :]
        y_train = y[:train_size]
        x_valid = X[train_size:, :]
        y_valid = y[train_size:]

        mean_x = x_train.mean()
        std_x = x_train.std()

        mean_y = y_train.mean()
        std_y = y_train.std()

        return MusicDataset(x_train, y_train, mean_x, std_x, mean_y, std_y), MusicDataset(
            x_valid, y_valid, mean_x, std_x, mean_y, std_y
        )
