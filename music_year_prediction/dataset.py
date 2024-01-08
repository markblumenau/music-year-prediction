import dvc.api
import pandas as pd
import torch


class MusicDataset(torch.utils.data.Dataset):
    def __init__(
        self, features, target, mean_features, std_features, mean_target, std_target
    ):
        self.features = features
        self.target = target
        self.mean_features = mean_features
        self.std_features = std_features
        self.mean_target = mean_target
        self.std_target = std_target
        self.features_count = features.shape[1]
        self.out = 1

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return {
            "features": torch.tensor(self.features[index, :]).float(),
            "target": torch.tensor(self.target[index]).float(),
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

        mean_features = features_train.mean()
        std_features = features_train.std()

        mean_target = target_train.mean()
        std_target = target_train.std()

        return MusicDataset(
            features_train,
            target_train,
            mean_features,
            std_features,
            mean_target,
            std_target,
        ), MusicDataset(
            features_valid,
            target_valid,
            mean_features,
            std_features,
            mean_target,
            std_target,
        )
