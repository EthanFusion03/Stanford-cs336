import torch
import math
from torch import Tensor
from jaxtyping import Float
from einops import einsum
from cs336_basics.transformer.softmax import softmax

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