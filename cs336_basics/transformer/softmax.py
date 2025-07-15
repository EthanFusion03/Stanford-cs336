import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    numerator =  torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / denominator
