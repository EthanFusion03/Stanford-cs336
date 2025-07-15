import torch
import math
from torch import Tensor
from jaxtyping import Float
from einops import einsum, rearrange
from cs336_basics.transformer.softmax import softmax
from cs336_basics.transformer.linear import Linear
from cs336_basics.transformer.rope import RotaryPositionalEmbedding

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # d_k = len(Q[...,-1]) This is wrong bc it only returns the first dim len
    d_k = Q.size(-1)
    scores = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k) # dim: (# of queries, # of keys) / (n, m)
    # Apply the mask before softmax
    if mask is not None:
        scores = scores.masked_fill(mask == False, -torch.inf)
    masked_weights = softmax(scores, dim=-1) # dim: (n, m)
    return einsum(masked_weights, V, "... n m, ... m d_v -> ... n d_v")

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int,
            use_rope: bool,
            max_seq_len: int = 2048, 
            theta: int = 10000.0,
            *args, 
            **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads # d_k = d_v = d_model / h

        # Projection matrices for queries, keys, and values
        self.query = Linear(d_model, d_model)
        self.key = Linear(d_model, d_model)
        self.value = Linear(d_model, d_model)

        # Final linear layer after concatenating heads
        self.output_layer = Linear(d_model, d_model)

        self.use_rope = use_rope
        if use_rope:
            # Establish RoPE
            self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len)

    def forward(self, x, token_positions: torch.Tensor | None = None, use_mask: bool = True):
        *batch, seq_len, d_model = x.size()

        if use_mask == True:
            # Setup causal/decoder masking
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            # Invert the mask matrix to fit our scaled_dot_product_attention
            mask = ~mask

        # Calculate projected x
        qx, kx, vx = self.query(x), self.key(x), self.value(x)

        # Split queries, keys, and values into multiple heads
        qx = rearrange(qx, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads = self.num_heads, head_dim = self.head_dim)
        kx = rearrange(kx, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads = self.num_heads, head_dim = self.head_dim)
        vx = rearrange(vx, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads = self.num_heads, head_dim = self.head_dim)

        if self.use_rope:
            # Calculate positional encoding for queries and keys
            if token_positions is None:
                token_positions = torch.arange(seq_len)
            qx = self.rope(qx, token_positions)
            kx = self.rope(kx, token_positions)

        # Calculate attention
        attention = scaled_dot_product_attention(qx, kx, vx, mask)
        attention = rearrange(attention, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)")
        attention = self.output_layer(attention)

        return attention