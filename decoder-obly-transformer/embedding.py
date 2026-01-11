import torch
import torch.nn as nn
import math

class ModelArgs:
    def __init__(self):
        self.vocab_size = 50257
        self.d_model = 768
        self.max_seq_len = 1024   # Context length
        self.dropout = 0.1

class TransformerEmbeddings(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.token_embedding = nn.Embedding(args.vocab_size, args.d_model)

        #pos embeddings: Look up vectors for positions 0, 1, 2....
        self.position_embedding = nn.Embedding(args.max_seq_len, args.d_model)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        B, T = x.shape   # x shape: (batch_size, seq_len)

        positions = torch.arange(0, T, device=x.device)
        
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)

        x = tok_emb + pos_emb

        return self.dropout(x)
    
args = ModelArgs()
emb_layer = TransformerEmbeddings(args)

test_input = torch.randint(0, args.vocab_size, (2, 10))
output = emb_layer(test_input)

print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")