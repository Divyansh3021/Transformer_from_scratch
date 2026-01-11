import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelArgs:
    def __init__(self):
        self.vocab_size = 50257
        self.d_model = 768
        self.max_seq_len = 1024   # Context length
        self.dropout = 0.1
        self.num_heads = 1

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

class CausalSelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.d_model % args.num_heads == 0, "d_model must be a multiple of num_heads"

        self.d_head = args.d_model // args.num_heads
        self.num_heads = args.num_heads

        self.c_attn = nn.Linear(args.d_model, 3 * args.d_model)    # Projections of Q K V

        self.c_proj = nn.Linear(args.d_model, args.d_model)    # Output projection to mix the heads back together

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

        # 3. The causal mask (buffer prevents it from being updated as a parameter)
        bias = torch.tril(torch.ones(args.max_seq_len, args.max_seq_len))
        self.register_buffer("bias", bias.view(1, 1, args.max_seq_len, args.max_seq_len))


    def forward(self, x):
        B, T, C = x.size() # Batch, Seq len, d_model

        # A: Calculate Q, K, V
        qkv = self.c_attn(x)

        q, k, v = qkv.split(C, dim=2)

        # B: Reshape for multi-head from (B, T, C) -> (B, T, num_heads, d_head)
        # then transpose to (B, num_heads, T, d_head) for matrix multiplication
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        # scaled dot product attention
        # (B, nh, T, dh) @ (B, nh, dh, T) -> (B, nh, T, T)
        att = (q @ k.transpose(2, 3)) * (1.0 / math.sqrt(k.size(-1)))
        print(f"k size: {k.size}")

        # D: apply mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # E: aggregate values
        # (B, nh, T, T) @ (B, nh, T, dh) -> (B, nh, T, dh)
        y = att @ v

        # Reassemble values
        # (B, T, nh, dh) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B,T,C)

        return self.resid_dropout(self.c_proj(y))
    

class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.c_fc = nn.Linear(args.d_model, 4 * args.d_model)
        self.c_proj = nn.Linear(4 * args.d_model, args.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        # X shape: (Batch, seq_len, d_model)

        #Project up
        x = self.c_fc(x)

        #apply non-linearity
        x = self.act(x)

        #project down
        x = self.c_proj(x)

        x = self.dropout(x)
        return x
    

class Block(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.ln_1 = nn.LayerNorm(args.d_model)
        self.ln_2 = nn.LayerNorm(args.d_model)

        self.attn = CausalSelfAttention(args)

        self.mlp = MLP(args)

    def forward(self, x):
        # Pre-Norm
        x = x + self.attn(self.ln_1(x))

        x = x + self.mlp(self.ln_2(x))

        return x

## Logic
# logits = LM_Head(LayerNorm(Blocks(Embed(x))))

class GPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        #1. Embeddings 
        self.transformer_embeddings = TransformerEmbeddings(self.args)

        #2. Stack of Blocks
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.n_layer)])

        #3. Final Layer Norm
        self.ln_f = nn.LayerNorm(args.d_model)

        #4. lm_head (Output Projection)
        # Maps final vectors to probabilities over the vocabulary
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

        # Weight typing: Link the weights of the embedding and the output head
        # This means training one updates the other
        self.transformer_embeddings.token_embedding.weight = self.lm_head.weight

        # Initialize weights (Crucial for deep transformers!)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Normal distribution initialization is standard for Transformers
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: (Batch, Seq_len)

        # 1. Embeddings
        x = self.transformer_embeddings(idx)

        # 2. Run through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # 3. Layer Normalization (Final)
        x = self.ln_f(x)

        # 4. Projecting logits
        logits = self.lm_head(x)  #Shape: (Batch, Seq_len, Vocab_size)

        loss = None
        if targets is not None:
            # We flatten the batch and sequence dimensions for CrossEntropyLoss
            # logits view: (Batch * Seq_len, Vocab_size)
            # targets view: (Batch * Seq_len)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        #idx: (batch, seq_len)
        for _ in range(max_new_tokens):
            # crop context if needed
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]

            # 2. Forward Pass
            logits, _ = self(idx_cond)

            # 3. Focus on last time step
            logits = logits[:, -1, :]

            # 4. Apply temperature
            logits = logits / temperature

            # 5. top-k sampling
            if top_k is not None:
                V, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < V[:, [-1]]] = -float("Inf")

            # 6. Calc. Probabilities
            probs = F.softmax(logits, dim=-1)  # (Batch, Vocab_size)

            # 7. Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # 8. Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, seq_len+1)

        return idx
    

device = "cuda" if torch.cuda.is_available() else "cpu"
args = ModelArgs()
args.n_layer = 4
args.num_heads = 4
args.d_model = 256

model = GPT(args).to(device)
model.eval()

print(f"Using device: {device}")


context = torch.zeros((1,1), dtype=torch.long, device=device)

print("Generating...")
generated_ids = model.generate(context, max_new_tokens=20)

print(f"Generated token IDs: {generated_ids.tolist()[0]}")

# print(f"Model built! Parameter count: {sum(p.numel() for p in model.parameters())}")

# # Test forward pass
# dummy_input = torch.randint(0, args.vocab_size, (2, 64))   # Batch=2, Seq_len = 64
# logits, _ = model(dummy_input)
# print(f"Output Logits shape: {logits.shape}")