import torch
from torch import nn


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask):
        """

        Args:
            q (torch.Tensor): [Batch size x Head  x sequence dimension x key dimension]
            k (torch.Tensor): [Batch size x Head x sequence dimension x key dimension]
            v (torch.Tensor): [Batch size x Head x sequence dimension x key dimension]
            mask (_type_): [Batch size, 1, 1, Sequence length] or
            [Batch size, 1, Sequence length, Sequence length]. Defaults to None.

        Returns:
            atatention: [Batch size x Head x sequence dimension x sequence dimension]
        """

        d_key = k.size(3)
        attention = (q @ k.transpose(2, 3)) / torch.sqrt(d_key)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -float("inf"))

        score = self.softmax(attention)
        return score @ v
