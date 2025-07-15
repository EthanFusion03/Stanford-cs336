import torch
from einops import einsum, rearrange
from torch import sin, cos

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
            self, 
            theta: float, 
            d_k: int, 
            max_seq_len: int, 
            device: torch.device | None = None
        ):
        super().__init__()

        """The rotation angles θ only depend on the token's position (i) 
        and the dimension's index (k), not on the input vector itself. 
        Therefore, the sin(θ) and cos(θ) values can be calculated once and reused for every forward pass."""
        
        # Calculate a table of sine and cosine values for every position up to the model's maximum sequence length.
        dim_indices = torch.arange(0, d_k, 2, device=device)
        inv_freq = 1.0 / (theta ** (dim_indices / d_k))
        # inv_freq dim: (d_k/2)

        sq_indices = torch.arange(0, max_seq_len, device=device)
        angles = einsum(sq_indices, inv_freq, "i, j -> i j")
        # angles have dim: (max_seq_len, d_k/2)

        sin_vals = sin(angles)
        cos_vals = cos(angles)
        
        # Save this table as a non-persistent buffer
        self.register_buffer("sin_vals", sin_vals, persistent=False)
        self.register_buffer("cos_vals", cos_vals, persistent=False)
        # Both have dim: (max_seq_len, d_k/2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. """
        # Step 1: Get sin and cos values for the given token positions.
        # token_positions has shape (seq_len,)
        # self.sin_vals has shape (max_seq_len, d_k / 2)
        sin = self.sin_vals[token_positions] # Shape: (seq_len, d_k / 2)
        cos = self.cos_vals[token_positions] # Shape: (seq_len, d_k / 2)

        # Step 2: Prepare sin and cos for broadcasting by duplicating each value.
        # This makes their shape compatible with the input tensor x.
        sin = sin.repeat_interleave(2, dim=-1) # Shape: (seq_len, d_k)
        cos = cos.repeat_interleave(2, dim=-1) # Shape: (seq_len, d_k)

        # The rotation for a single pair [k₁, k₂] is: k'₁ = k₁cos(θ) - k₂sin(θ), k'₂ = k₁sin(θ) + k₂cos(θ)
        k_paired = rearrange(x, '... (d two) -> ... d two', two=2) # (..., seq_len, d_k, 2)
        k1 = k_paired[..., 0]
        k2 = k_paired[..., 1]

        x_rotated_90_paired = torch.stack([-k2, k1], dim=-1)
        x_rotated_90 = rearrange(x_rotated_90_paired, '... d two -> ... (d two)', two=2)

        return x * cos + x_rotated_90 * sin

        

