from torch import nn

class PositionalWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, drop_prob):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_prob, inplace=True)
        self.linear2 = nn.Linear(d_hidden, d_model)
        
    def forward(self, x):
        """

        Args:
            x (torch.Tenspr): [Batch size x seq dim x d model]
            
        Returns:
            torch.Tensor: [Batch size x seq dim x d model]
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear2(x)