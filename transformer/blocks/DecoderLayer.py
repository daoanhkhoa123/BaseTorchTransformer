from transformer.layers.MultiHeadAttention import MultiHeadAttention
from transformer.layers.LayerNorm import LayerNorm
from transformer.layers.PositionalWiseFeedForward import PositionalWiseFeedForward
from torch import nn

EPS = 0.001


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_hidden, drop_prob):
        super().__init__()

        self.sel_attention = MultiHeadAttention(n_heads, d_model)
        self.drop_out1 = nn.Dropout(drop_prob, inplace=True)
        self.layer_norm1 = LayerNorm(d_model, eps=EPS)

        self.decode_encode_attention = MultiHeadAttention(n_heads, d_model)
        self.drop_out2 = nn.Dropout(drop_prob, inplace=True)
        self.layer_norm2 = LayerNorm(d_model, eps=EPS)

        self.pos_feedforward = PositionalWiseFeedForward(
            d_model, d_hidden, drop_prob)
        self.drop_out3 = nn.Dropout(drop_prob, inplace=True)
        self.layer_norm3 = LayerNorm(d_model, eps=EPS)

    def forward(self, dec, enc, trg_mask, src_mask):
        """

        Args:
            enc : [Batch size x seq dim x model dim]
            dec : [Batch size x seq dim x model dim]
            trg_mask : [Batch size x 1 x seq dim x seq dim]
            src_mask : [Batch size x 1 x seq dim x 1]

        Returns:
            [Batch size x seq dim x model dim]
        """

        # target mask first, then source (for padding)
        dec_attn = self.sel_attention(dec, dec,  dec, trg_mask)
        dec_attn = self.drop_out1(dec_attn)
        dec_attn = self.layer_norm1(dec_attn+dec)
        
        enc_attn = dec_attn
        if enc is not None:
            enc_attn = self.decode_encode_attention(dec_attn, enc, enc, src_mask)
            enc_attn = self.drop_out2(enc_attn)
            enc_attn = self.layer_norm2(enc_attn + dec_attn)
            
        pos_ff = self.pos_feedforward(enc_attn)
        pos_ff = self.drop_out3(pos_ff)
        pos_ff = self.layer_norm3(pos_ff + enc_attn)
        
        return pos_ff