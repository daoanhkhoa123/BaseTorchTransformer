import torch
from torch import nn

TEN_THOUSAND = 10000

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_seq_dim:int, device):
        super().__init__()
        
        _2i = torch.arange(0, d_model, 2, device= device)
        pos = torch.arange(0, max_seq_dim, device= device).unsqueeze(1)
        freq_seq = pos / torch.pow(TEN_THOUSAND, _2i/d_model)
        
        self.encoding  = torch.empty(max_seq_dim, d_model, device=device, requires_grad=False)
        self.encoding [:, 0::2] = torch.sin(freq_seq)
        self.encoding [:, 1::2] = torch.cos(freq_seq)
        
    def forward(self, x:torch.Tensor)        :
        """

        Args:
            x (torch.Tensor): [Batch size x Sequence length]
            
        Returns:
            positional encoding : [Batch size x sequence length x model dimension]
        """
        
        seq_length = x.size(1)
        
        return self.encoding[:seq_length, :]