from einops import einsum
import torch
import math

class Linear(torch.nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        # 3. Wrap the initialized tensor in nn.Parameter.
        self.w = torch.nn.Parameter(self.init_linear_weight())
    
    def init_linear_weight(self):
        # 1. Create an empty tensor with the correct shape, device, and dtype.
        weight = torch.empty((self.out_features, self.in_features), device=self.device, dtype=self.dtype)
        mu = 0.0
        # 2. Calculate initialization parameters and initialize the tensor in-place.
        sigma = math.sqrt(2 / (self.in_features + self.out_features))
        weight = torch.nn.init.trunc_normal_(weight, mean=mu, std=sigma, a=-3*sigma, b=3*sigma)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.w, "... in_feature, out_feature in_feature -> ... out_feature")
