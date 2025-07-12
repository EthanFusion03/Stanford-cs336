import torch
from einops import einsum

class Rmsnorm(torch.nn.Module):
    def __init__(
            self, 
            d_model: int, 
            eps: float = 1e-5, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
        ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = torch.nn.Parameter(torch.ones(self.d_model, device=self.device, dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length, d_model) -> 
        # output: (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Step 1: find RMS(a)
        x_squared = x.pow(2) # (batch_size, sequence_length, d_model)
        mean_sq = x_squared.mean(dim=-1, keepdim=True) # (batch_size, sequence_length, 1)
        rms_a = torch.sqrt(mean_sq + self.eps) # (batch_size, sequence_length, 1)

        # Step 2: find RMSNorm(a_i)
        xg = einsum(x, self.g, "bs sl d_model, d_model -> bs sl d_model")
        RMSNorm = (xg / rms_a).to(in_dtype)
        
        return RMSNorm 