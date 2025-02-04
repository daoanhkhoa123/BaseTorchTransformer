from transformer.blocks.DecoderLayer import DecoderLayer
from transformer.embeddings.TransformerEmbedding import TransformerEmbedding 
from torch import nn

class Decoder(nn.Module):
    def __init__(self, n_layers, vocab_size, d_model, max_seq_len, n_heads, d_hidden, drop_prob, decode_vocab_size, device=None):
        super().__init__()
        
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_seq_len, drop_prob, device)
        self.decoder_layers = nn.ModuleList([DecoderLayer(n_heads, d_model, d_hidden, drop_prob)
                                             for _ in range(n_layers)])
        
        self.linear = nn.Linear(d_model, decode_vocab_size)
        
    def forward(self, dec, enc, trg_mask, src_mask):
        """

        Args:
            enc (encoder source): [batch size x seq dim]
            dec (target): [batch size x seq dim]
            trg_mask : [batch size x 1 x seq dim x seq dim]
            src_mask : [batch size x 1 x seq dim x 1]
            
        Returns:
        [batch size x seq dim x decode vocab size]
        """
        
        embedding = self.embedding(dec) # embedding is our target
        for decoders in self.decoder_layers:
            embedding = decoders(embedding, enc,trg_mask, src_mask)
            
        return self.linear(embedding)
        
        