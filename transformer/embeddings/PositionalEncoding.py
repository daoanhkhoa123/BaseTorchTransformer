from torch import nn
import torch

TEN_THOUSAND = 10000


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, device=None):
        super().__init__()

        pos = torch.arange(0, max_seq_len, device=device).unsqueeze_(1)
        _2i = torch.arange(0, max_seq_len, 2, device=device)
        freqency = pos / torch.pow(TEN_THOUSAND, _2i/d_model)

        self.embedding = torch.empty(
            max_seq_len, d_model, requires_grad=False, device=device)
        self.embedding[:, 0::2] = torch.sin(freqency)
        self.embedding[:, 1::2] = torch.cos(freqency)

    def forward(self, x: torch.Tensor):
        """

        Args:
            x (torch.Tensor): [Batch size x seq len]

        Returns:
            (torch.Tensor): [Seq len x d model]
        """

        seq_len = x.size(1)
        return self.embedding[:seq_len, :]
