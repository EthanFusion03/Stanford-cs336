import torch

def gradient_clipping(params: list, max_l2_norm: float) -> None:
    grads = [p.grad.detach().flatten() for p in params if p.grad is not None]

    # If no parameters have gradients, end the func
    if len(grads) == 0:
        return
    
    # 1. Flatten all gradients and concatenate them into a single tensor
    grads = torch.cat(grads)
    # 2. Now, compute the norm on this single, combined tensor
    grads_l2 = torch.norm(grads, p=2)
    eps = 1e-6
    if grads_l2 >= max_l2_norm:
        clip_coef = max_l2_norm / (grads_l2 + eps)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(clip_coef)