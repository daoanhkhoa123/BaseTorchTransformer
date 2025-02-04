from Encoder import Encoder
from Decoder import Decoder
from torch import nn
import torch


class Transformer(nn.Module):
    def __init__(self, enc_pad_idx, dec_pad_idx, dec_sos_idx, n_layers, vocab_size, d_model, max_seq_len, n_heads, d_hidden, drop_prob, decode_vocab_size, device=None):
        super().__init__()
        self.enc_pad_idx = enc_pad_idx
        self.dec_pad_idx = dec_pad_idx
        self.dec_sos_idx = dec_sos_idx
        self.device = device

        self.encoder = Encoder(n_layers, vocab_size, d_model,
                               max_seq_len, n_heads, d_hidden, drop_prob, device)
        self.decoder = Decoder(n_layers, vocab_size, d_model, max_seq_len,
                               n_heads, d_hidden, drop_prob, decode_vocab_size, device=None)

    def forward(self, src, trg):
        """

        Args:
            src (Sequence): [batch size x source seq len]
            trg (Sequence): [batch size x target seq len]

        """

        src_mask = self.prepare_src_mask(src)
        trg_mask = self.prepare_trg_mask(trg)

        enc = self.encoder(src, src_mask)
        dec = self.decoder(trg, enc, trg_mask, src_mask)

        return dec

    def prepare_src_mask(self, x: torch.Tensor):
        """

        Args:
            x (Sequence): [batch size x seq len]

        Returns:
            mask: [batch size x 1 x seq len x 1]
        """
        return (x != self.enc_pad_idx).unsqueeze(1).unsqueeze(3)

    def prepare_trg_mask(self, x: torch.Tensor):
        """

        Args:
            x (Sequence): [batch size x seq len]

        Returns:
            mask: [batch size x 1 x seq len x seq len]
        """

        pad_mask: torch.Tensor = x != self.dec_pad_idx
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(3)

        seq_len = x.size(1)
        tril_mask = torch.tril(torch.ones(seq_len, seq_len,
                                          dtype=torch.ByteTensor, device=self.device))
        return pad_mask & tril_mask
