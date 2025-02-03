from models.layers.MultiHeadAttention import MultiHeadAttention
from models.layers.PositionwiseFeedForward import PositionwiseFeedForward
from models.layers.LayerNorm import LayerNorm
from models.embedding.TransformerEmbedding import TransformerEmbedding
from torch import nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head:int, hidden_dim:int, drop_p:int):
        super().__init__()
        
        self.multi_head_attention1 = MultiHeadAttention(d_model, n_head) # masked self-attention for decode
        self.dropout1 = nn.Dropout(drop_p, inplace=True)
        self.layer_norm1= LayerNorm(d_model)
                
        self.multi_head_attention2 = MultiHeadAttention(d_model, n_head) # encode self-attention for encode
        self.dropout2 = nn.Dropout(drop_p, inplace=True)
        self.layer_norm2 = LayerNorm(d_model)
        
        self.positionwise_feed_forward = PositionwiseFeedForward(d_model, hidden_dim, drop_p)
        self.dropout3 = nn.Dropout(drop_p, inplace=True)
        self.layer_norm3 = LayerNorm(d_model)
                
    def forward(self, decode, encode, trg_mask, src_mask):
        """

        Args:
            decode (torch.Tensor): [Batch size x sequence length x model dimension]
            encode (torch.Tensor): [Batch size x sequence length x model dimension]
            trg_mask (torch.Tensor): [batch_size, 1, sequence, sequence]
            src_mask (torch.Tensor): [batch_size, 1, 1, sequence]
            
        Returns:
            (torch.Tensor): [Batch size x sequence length x model dimension]
        """
        
        attention1 = self.multi_head_attention1(decode, decode, decode, trg_mask)
        attention1 = self.dropout1(attention1)
        attention1 = self.layer_norm1(attention1 + decode)
        
        if encode is not None:
            attention2 = self.multi_head_attention2(attention1, encode, encode, src_mask)
            attention2 = self.dropout2(attention2)
            attention2 = self.layer_norm2(attention2 + attention1)
            
        pos_ff = self.positionwise_feed_forward(attention2)
        pos_ff = self.dropout3(pos_ff)
        pos_ff = self.layer_norm3(pos_ff + attention2)
        
        return pos_ff

