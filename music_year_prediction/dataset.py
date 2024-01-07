import dvc.api
import pandas as pd
import torch


class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, features, target, mean_x, std_x, mean_y, std_y):
        self.features = features
        self.target = target
        self.mean_x = mean_x
        self.std_x = std_x
        self.mean_y = mean_y
        self.std_y = std_y
        self.features_count = features.shape[1]
        self.out = 1

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return {
            "x": torch.tensor(self.features[index, :]).float(),
            "y": torch.tensor(self.target[index]).float(),
        }


def make_datasets(train_size: int = 463715):
    with dvc.api.open("./data/YearPredictionMSD.txt") as file:
        df = pd.read_csv(file, header=None)
        features = df.iloc[:, 1:].values
        target = df.iloc[:, 0].values
        features_train = features[:train_size, :]
        target_train = target[:train_size]
        features_valid = features[train_size:, :]
        target_valid = target[train_size:]

        mean_x = features_train.mean()
        std_x = features_train.std()

        mean_y = target_train.mean()
        std_y = target_train.std()

        return MusicDataset(
            features_train, target_train, mean_x, std_x, mean_y, std_y
        ), MusicDataset(features_valid, target_valid, mean_x, std_x, mean_y, std_y)
