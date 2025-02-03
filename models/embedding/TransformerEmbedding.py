from PositionalEncoding import PositionalEncoding
from TokenEmbedding import TokenEmbedding
from torch import nn


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_dim, drop_p: float, device):
        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model, max_seq_dim, device)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        return self.dropout(self.token_embedding(x) + self.position_embedding(x))
