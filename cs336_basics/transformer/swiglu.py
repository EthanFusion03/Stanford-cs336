from jaxtyping import Float
from torch import sigmoid, Tensor 
from einops import einsum

def swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Step 1: w_1 * x
    w1x = einsum(in_features, w1_weight, "... d_model, d_ff d_model -> ... d_ff")
    # Step 2: SiLU(w1x)
    silu_w1x = w1x * sigmoid(w1x)
    # Step 3: w_3 * x
    w3x = einsum(in_features, w3_weight, "... d_model, d_ff d_model -> ... d_ff")
    # Step 4: element(w1x, w3x)
    element_mult = silu_w1x * w3x
    # Step 5: FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)
    output = einsum(element_mult, w2_weight, "... d_ff, d_model d_ff -> ... d_model")
    return output