import torch

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)


        self.stoi = {ch:i for i, ch in enumerate(chars)}
        self.itos = {i:ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    def decode(self, l):
        if isinstance(l, torch.Tensor):
            l = l.tolist()
        return "".join([self.itos[i] for i in l])
    
# text_data = "Hello World! This is a simple tokenizer."
# tokenizer = CharTokenizer(text_data)

# print(f"Vocab Size: {tokenizer.vocab_size}")
# print(f"Mapping: {tokenizer.stoi}")

# # Test Encoding
# encoded = tokenizer.encode("Hello")
# print(f"Encoded 'Hello': {encoded}")

# # Test Decoding
# decoded = tokenizer.decode(encoded)
# print(f"Decoded: {decoded}")