from transformer.blocks.EncoderLayer import EncoderLayer
from transformer.embeddings.TransformerEmbedding import TransformerEmbedding
from torch import nn

class Encoder(nn.Module):
    def __init__(self, n_layers, vocab_size, d_model, max_seq_len, n_heads, d_hidden, drop_prob, device=None):
        super().__init__()
        
        self.embedding = TransformerEmbedding(vocab_size, d_model,max_seq_len, drop_prob, device)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model,n_heads,d_hidden,drop_prob)
                                             for _ in range(n_layers)])
        
    def forward(self, x, src_mask):
        """

        Args:
            x: [Batch size x seq len]
            
        Returns:
            [Batch size x seq len x model]
        """
        embedding = self.embedding(x)
        for encoder in self.encoder_layers:
            embedding = encoder(embedding, src_mask)
            
        return embedding
        
        