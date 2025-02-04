import torch
from torch import nn


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask):
        """

        Args:
            q (torch.Tensor): [batch size x n head x seq dim x key dim]
            k (torch.Tensor): [batch size x n head x seq dim x key dim]
            v (torch.Tensor): [batch size x n head x seq dim x key dim]
            mask (torch.Tensor): [batch size x 1 x seq dim x seq dim] (target mask) or
                                [batch size x 1 x 1 x seq dim] (source mask)

        Returns:
            torch.Tensor: [batch size x 1 x seq dim x seq dim] (target) or
                                [batch size x 1 x 1 x seq dim] (source)
        """

        d_key = k.size(3)
        attention: torch.Tensor = q @ k.transpose(2, 3) / torch.sqrt(d_key)

        if mask is not None:
            attention = attention.masked_fill_(mask == 0, -float("inf"))

        attention = self.softmax(attention)
        return attention @ v
