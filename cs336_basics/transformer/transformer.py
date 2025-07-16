import torch
from cs336_basics.transformer.attention import MultiHeadSelfAttention
from cs336_basics.transformer.swiglu import swiglu


# A common use case in code
class prenorm_XformerBlock(torch.nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int,
            d_ff: int
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, use_rope=True)
        self.ffn = swiglu(d_model, d_ff)