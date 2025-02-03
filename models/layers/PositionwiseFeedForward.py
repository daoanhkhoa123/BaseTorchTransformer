from torch import nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, drop_p: float):
        super().__init__()

        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_p, inplace=True)
        self.linear2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x
