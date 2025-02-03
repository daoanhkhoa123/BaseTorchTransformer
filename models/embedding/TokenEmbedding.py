from torch import nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model, padding_idx=1):
        super().__init__(vocab_size, d_model, padding_idx)
