import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps):
        super().__init__()
        self.eps = eps
        
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros_like(self.gamma))
        
    def forward(self, x:torch.Tensor):
        """

        Args:
            x (torch.Tensor): [Batch size x seq length x d model]

        Returns:
            torch.Tensor: [Batch size x seq length x d model]
        """
        
        x -= x.mean(-1, keepdim=True)
        x /= torch.sqrt(x.var(-1,unbiased=False, keepdim=True) + self.eps)
        
        return self.gamma *x + self.beta