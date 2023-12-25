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
        output_size=1,
    ):
        super(LinearModel, self).__init__()
        self.input = LinearBlock(hidden_size, input_size)
        self.hidden_blocks = []
        for _i in range(block_count):
            self.hidden_blocks.append(LinearBlock(hidden_size))
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input(x)
        for i in range(len(self.hidden_blocks)):
            x = self.hidden_blocks[i](x)
        return self.output(x)
