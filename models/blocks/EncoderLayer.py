from torch import nn
from models.layers.MultiHeadAttention import MultiHeadAttention
from models.layers.PositionwiseFeedForward import PositionwiseFeedForward
from models.layers.LayerNorm import LayerNorm


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, hidden_dim: int, drop_p: float):
        super().__init__()

        self.encode_attention = MultiHeadAttention(d_model, n_head)
        self.dropout1 = nn.Dropout(drop_p, inplace=True)
        self.layernorm1 = LayerNorm(d_model)

        self.positionwise_feed_forward = PositionwiseFeedForward(
            d_model, hidden_dim, drop_p)
        self.dropout2 = nn.Dropout(drop_p, inplace=True)
        self.layernorm2 = LayerNorm(d_model)

    def forward(self, encode, src_mask):
        """
        Args:
            encode (torch.Tensor): [Batch size x sequence dimension x model dimension]
            src_mask (torch.Tensor): [Batch size, 1, 1, Sequence length] or
            [Batch size, 1, Sequence length, Sequence length] 

        Returns:
            torch.Tensor: [Batch size x sequence dimension x model dimension]
        """
        attention = self.encode_attention(encode, encode, encode, src_mask)
        attention = self.dropout1(attention)
        attention = self.layernorm1(attention + encode)

        pos_ff = self.positionwise_feed_forward(attention)
        pos_ff = self.dropout2(pos_ff)
        pos_ff = self.layernorm2(pos_ff + attention)

        return pos_ff
