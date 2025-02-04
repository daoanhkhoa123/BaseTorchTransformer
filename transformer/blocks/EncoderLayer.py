from transformer.layers.MultiHeadAttention import MultiHeadAttention
from transformer.layers.LayerNorm import LayerNorm
from transformer.layers.PositionalWiseFeedForward import PositionalWiseFeedForward
from torch import nn

EPS = 0.001


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_hidden, drop_prob):
        super().__init__()

        self.attention = MultiHeadAttention(n_heads, d_model)
        self.drop_out1 = nn.Dropout(drop_prob, inplace=True)
        self.layer_norm1 = LayerNorm(d_model, eps=EPS)

        self.pos_feedforward = PositionalWiseFeedForward(
            d_model, d_hidden, drop_prob)
        self.drop_out2 = nn.Dropout(drop_prob, inplace=True)
        self.layer_norm2 = LayerNorm(d_model, eps=EPS)

    def forward(self, enc, src_mask):
        """

        Args:
            enc: [Batch size x seq len x d model]
            dec : [Batch size x seq len x d model]
            src_mask : [Batch size x 1 x seq len x 1]

        Returns:
            [Batch size x seq len x d model]
        """
        attention = self.attention(enc, enc, enc, src_mask)
        attention = self.drop_out1(attention)
        attention = self.layer_norm1(enc + attention)

        pos_ff = self.pos_feedforward(attention)
        pos_ff = self.drop_out2(pos_ff)
        pos_ff = self.layer_norm2(pos_ff + attention)

        return pos_ff
