import torch


class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, mean_x, std_x, mean_y, std_y):
        self.x = x
        self.y = y
        self.mean_x = mean_x
        self.std_x = std_x
        self.mean_y = mean_y
        self.std_y = std_y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return {
            "x": torch.tensor(self.x[index, :]).float(),
            "y": torch.tensor(self.y[index]).float(),
        }
