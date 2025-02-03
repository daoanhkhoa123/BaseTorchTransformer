from Decoder import Decoder
from Encoder import Encoder
from torch import nn
import torch


class Transformer(nn.Module):
    def __init__(self, src_pad_idx: int, trg_pad_idx: int, trg_sos_idx: int,
                 n_layers: int, encode_vocab_size: int, d_model: int, max_seq_dim: int,
                 n_head: int, hidden_dim: int, drop_p: float, decode_vocab_size: int, device):
        super().__init__()
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx

        self.encoder = Encoder(n_layers, encode_vocab_size, d_model,
                               max_seq_dim, n_head, hidden_dim, drop_p, device)
        self.decoder = Decoder(n_layers, decode_vocab_size, d_model,
                               max_seq_dim, n_head, hidden_dim, drop_p, device)

    def forward(self, src: torch.Tensor, trg: torch.Tensor):
        trg_mask = self.prepare_trg_mask(trg)
        src_mask = self.prepare_src_mask(src)
        enc_src = self.encoder(src, trg_mask)
        return self.decoder(trg, enc_src, trg_mask, src_mask)

    def prepare_trg_mask(self, trg: torch.Tensor):
        """

        Args:
            trg (torch.Tensor): [Batch x target sequence length]

        Returns:
            torch.Tensor: [Batch x 1 x target sequence length x target sequence length]
        """
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)  # [batch x 1 x seq length x 1]
        trg_len = trg.size(1)

        # we could implement directly into the multi head attention
        tril = torch.tril(torch.ones(trg_len, trg_len,
                          dtype=torch.ByteTensor, device=self.device))
        return trg_pad_mask & tril

    def prepare_src_mask(self, src: torch.Tensor):
        """
        Args:
            src (torch.Tensor): [Batch size x Source equence length]

        Returns:
            torch.Tensor: [Batch size x 1 x 1 x Source sequence length]
        """
        # source pad mask
        # filter out pad token index
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
