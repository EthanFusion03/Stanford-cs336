import torch
from einops import einsum

class Embedding(torch.nn.Module):
    def __init__(
            self, 
            num_embeddings: int, 
            embedding_dim: int, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
        ):
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim 
        self.device = device
        self.dtype = dtype
        # Initialize the embedding lookup table
        self.emb_table = torch.nn.Parameter(self.init_embedding_table())
    
    def init_embedding_table(self):
        emb_table = torch.empty((self.vocab_size, self.d_model), device=self.device, dtype=self.dtype)
        mu = 0
        sigma = 1
        emb_table = torch.nn.init.trunc_normal_(emb_table, mu, sigma, -3, 3)
        return emb_table
    
    # x has dim = (batch_size, sequence_length)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb_table[x]