from torch import nn 
import torch

class LayerNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-6):
        super().__init__()
        
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros_like(self.gamma))
        self.eps = eps
        
    def forward(self, x:torch.Tensor):
        """

        Args:
            x (torch.Tensor):  [Batch size x key dimension x model dimension]

        Returns:
            torch.Tensor: [Batch size x key dimension x model dimension]
        """
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False,keepdim=True)
        
        x -= mean
        x /= torch.sqrt(var + self.eps)
        
        return self.gamma * x + self.beta
    