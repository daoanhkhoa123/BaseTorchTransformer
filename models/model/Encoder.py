from models.blocks.EncoderLayer import EncoderLayer
from models.embedding.TransformerEmbedding import TransformerEmbedding
from torch import nn


class Encoder(nn.Module):
    def __init__(self, n_layers: int, enc_vocab_size: int, d_model: int, max_seq_dim: int, n_head, hidden_dim, drop_p: float, device):
        super().__init__()

        self.transformer_embedding = TransformerEmbedding(
            enc_vocab_size, d_model, max_seq_dim, drop_p, device)
        self.encode_blocks = nn.ModuleList([EncoderLayer(d_model, n_head, hidden_dim, drop_p)
                                            for _ in range(n_layers)])

    def forward(self, src, trg_mask):
        """

        Args:
            src (torch.Tensor): [Batch size x sequence dimension x model dimension]
            trg_mask (torch.Tensor): [Batch size x sequence dimension x model dimension]

        Returns:
            (torch.Tensor): [Batch size x sequence dimension x model dimension]
        """
        
        embedding = self.transformer_embedding(src)
        for block in self.encode_blocks:
            embedding = block(embedding, trg_mask)

        return embedding
