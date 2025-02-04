from torch import nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model, device=None):
        super().__init__(vocab_size, d_model, 1, device)
