from ScaleDotProductAttention import ScaleDotProductAttention
import torch
from torch import nn


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_key = d_model//n_head

        self.w_q = nn.Linear(d_model, n_head * self.d_key,
                             bias=False)  # d model, d model
        self.w_k = nn.Linear(d_model, n_head * self.d_key, bias=False)
        self.w_v = nn.Linear(d_model, n_head * self.d_key, bias=False)
        # after this, we split result tensor into [batch size x head x seq dim x key dim]

        self.attention = ScaleDotProductAttention()
        # now is [batch size x head x seq dim x seq dim]

        self.linear = nn.Linear(d_model, d_model)  # after concat

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """

        Args:
            q (torch.Tensor): [Batch size x sequence length x model dimension]
            k (torch.Tensor): [Batch size x sequence length x model dimension]
            v (torch.Tensor): [Batch size x sequence length x model dimension]
            mask (torch.Tensor, optional): [Batch size, 1, 1, Sequence length] or
            [Batch size, 1, Sequence length, Sequence length]. Defaults to None.

        Returns:
            score (torch.Tensor): [Batch size x key dimension x model dimension]

        Note: Usually d_k = d_v = d_model // head
        """

        q = self.w_q(q)  # [Batch size x d model]
        k = self.w_k(k)
        v = self.w_v(v)

        batch_size = k.size(0)
        seq_length = k.size(1)
        q.view(batch_size, seq_length, self.n_head, self.d_key).transpose(1, 2)
        k.view(batch_size, seq_length, self.n_head, self.d_key).transpose(1, 2)
        v.view(batch_size, self.d_key, self.n_head, self.d_key).transpose(1, 2)

        # score shape is [batch_size, self.d_key, self.n_head, self.d_key]
        score: torch.Tensor = self.attention(q, k, v, mask)

        # concat
        score = score.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        return self.linear(score)
