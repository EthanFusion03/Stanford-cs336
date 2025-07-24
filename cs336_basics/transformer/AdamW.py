from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-2):
        defaults = {
            "lr": lr,
            "beta": betas,
            "eps": eps,
            "lam": weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["beta"]
            eps = group["eps"]
            lam = group["lam"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                # State initialization
                if len(state) == 0:
                    state["t"] = 1
                    # 1st moment vector (m)
                    state["m"] = torch.zeros_like(p.data)
                    # 2nd moment vector (v)
                    state["v"] = torch.zeros_like(p.data)
                
                m, v = state["m"], state["v"]
                t = state["t"]

                grad = p.grad.data # Get the gradient of loss with respect to p.
                # Update the 1st moment estimate
                # m = b1 * m + (1 - b1) * grad 
                # state["m"] = m
                m.mul_(b1).add_(grad, alpha=1-b1)

                # # Update the 2nd moment estimate
                # v = b2 * v + (1 - b2) * grad**2
                # state["v"] = v
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)

                # Compute adjusted lr for iteration t
                lr_t = (lr * math.sqrt(1 - b2**t)) / (1 - b1**t)
                # p.data -= lr_t * m / torch.sqrt(v) + eps # Update weight tensor in-place.
                p.data.addcdiv_(m, torch.sqrt(v).add_(eps), value= -lr_t)
                # p.data -= lr * lam * p.data
                p.data.add_(p.data, alpha=-lr*lam)

                state["t"] += 1

        return loss


def main():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = AdamW([weights], lr=1e-3)
    for t in range(100):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.

if __name__ == "main":
    main()