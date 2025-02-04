from PositionalEncoding import PositionalEncoding
from TokenEmbedding import TokenEmbedding
from torch import nn


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, drop_prob, device=None):
        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size, d_model, device)
        self.positional_embedding = PositionalEncoding(d_model, max_seq_len, device)
        self.drop_out = nn.Dropout(drop_prob, inplace=True)

    def forward(self, x):
        """
        Args:
            x : [Batch size x seq len]

        Returns:
            [Batch size x seq len x d model]
        """
        tok_embedding = self.token_embedding(x)
        pos_embedding = self.positional_embedding(x)
        return self.drop_out(tok_embedding+pos_embedding)
