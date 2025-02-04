from transformer.layers.ScaleDotProductAttention import ScaleDotProductAttention
from torch import nn
from torch import Tensor as torchTensor

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads:int, d_model:int):
        """
        Assume d_key = d_value = d_model // n_heads
        """
        super().__init__()
        self.n_heads = n_heads
        self.d_key = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        self.attention = ScaleDotProductAttention()
        self.linear = nn.Linear(n_heads//d_model, d_model) # self.d_key, d_model
        
    def forward(self, q:torchTensor,k:torchTensor,v:torchTensor, mask=None):
        """

        Args:
            q (torchTensor): [Batch size x seq dim x d model]
            k (torchTensor): [Batch size x seq dim x d model]
            v (torchTensor): [Batch size x seq dim x d model]
            mask (torch.Tensor, optional): [Batch size x 1 x seq dim x seq dim]. Defaults to None.
            
        Returns:
            torch.Tensor: [Batch size x seq dim x d model]
        """
        
        batch_size, seq_dim, d_model = k.size()
        
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        
        # [batch size x n head x seq dim x d key]
        q = q.view(batch_size, seq_dim, self.n_heads,self.d_key).transpose(1,2)
        k = k.view(batch_size, seq_dim, self.n_heads,self.d_key).transpose(1,2)
        v = v.view(batch_size, seq_dim, self.n_heads,self.d_key).transpose(1,2)
        
        attention:torchTensor = self.attention(q,k,v, mask)
        attention = attention.transpose(1,2).contiguous().view(batch_size, seq_dim, d_model)
        
        return self.linear(attention)