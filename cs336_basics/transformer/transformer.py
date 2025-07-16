import torch
from cs336_basics.transformer.attention import MultiHeadSelfAttention
from cs336_basics.transformer.swiglu import swiglu
from cs336_basics.transformer.rmsnorm import Rmsnorm


# A common use case in code
class prenorm_XformerBlock(torch.nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int,
            d_ff: int,
            use_rope: bool = True,
            max_seq_len: int = 2048, 
            theta: int = 10000.0,
    ):
        super().__init__()
        self.attn_layer = MultiHeadSelfAttention(d_model, num_heads, use_rope=use_rope, max_seq_len=max_seq_len, theta=theta)
        self.ffn_layer = swiglu(d_model, d_ff)
        self.norm1 = Rmsnorm(d_model)
        self.norm2 = Rmsnorm(d_model)
    
    # x has input dim: (batch_size, seq_len, d_model)
    def forward(self, x):
        # Attention prcoess step
        x = x + self.attn_layer(self.norm1(x))
        # FFN step
        x = x + self.ffn_layer(self.norm2(x))
        return x