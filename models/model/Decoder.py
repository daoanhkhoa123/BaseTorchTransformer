from models.blocks.DecoderLayer import DecoderLayer
from models.embedding import TransformerEmbedding
from torch import nn


class Decoder(nn.Module):
    def __init__(self, n_layers: int, decode_vocab_size: int, d_model: int, max_seq_dim: int,  n_head: int, hidden_dim: int, drop_p: int,  device):
        super().__init__()

        self.transformer_embedding = TransformerEmbedding(
            decode_vocab_size, d_model, max_seq_dim, drop_p, device)
        self.decoder_blocks = nn.ModuleList([DecoderLayer(d_model, n_head, hidden_dim, drop_p)
                                             for _ in range(n_layers)])

        # softmax is usually put at the loss calculation step
        # specifically with cross entropy, which calcuate softmax internally
        self.linear = nn.Linear(d_model, decode_vocab_size)

    def forward(self, trg, encode_src, trg_mask, src_mask):
        """

        Args:
            trg (torch.Tensor): [Batch size x target sequence dimension]
            encode_src (torch.Tensor): [Batch size x source sequence dimension x model dimension]
            trg_mask (torch.Tensor): [Batch size x 1 x target sequence dimension x target sequence dimension]
            src_mask (torch.Tensor): [Batch size x 1 x 1 x source sequence dimension]

        Returns:
            (torch.Tensor): [Batch size x target sequence dimension x decode3 vocab size]
        """
        trg = self.transformer_embedding(trg)
        for block in self.decoder_blocks:
            trg = block(trg, encode_src, trg_mask, src_mask)

        trg = self.linear(trg)
        return trg
