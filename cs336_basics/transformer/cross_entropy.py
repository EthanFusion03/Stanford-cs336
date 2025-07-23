import torch

def perplexity(logits: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
    """
    Computes the average cross-entropy loss based on the formula:
    l_i = -log_softmax(o_i)[x_{i+1}]
    """
    stable_logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    log_sum_exp = torch.log(torch.sum(torch.exp(stable_logits), dim=-1, keepdim=True))
    log_probs = stable_logits - log_sum_exp
    target_log_probs = torch.gather(log_probs, dim=-1, index=x_target.unsqueeze(-1))
    loss_per_example = -target_log_probs.squeeze(-1)

    # First, find the single average cross-entropy loss over the whole batch.
    # By omitting the 'dim' argument, .mean() averages over all elements.
    # Then, calculate perplexity by exponentiating the average loss.
    perplexity = torch.exp(torch.mean(loss_per_example))

    return perplexity

def cross_entropy(logits: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
    """
    Computes the average cross-entropy loss based on the formula:
    l_i = -log_softmax(o_i)[x_{i+1}]
    """
    
    # Formula: This step implements the log-sum-exp trick for numerical stability.
    # It's equivalent to calculating o_i - c, where c = max(o_i).
    # This prevents overflow when exp() is called on large logit values.
    stable_logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    
    # Formula: This calculates the log of the partition function (the denominator of softmax).
    # It computes log(sum(exp(o_i[a]))) for all 'a' in the vocabulary.
    log_sum_exp = torch.log(torch.sum(torch.exp(stable_logits), dim=-1, keepdim=True))
    
    # Formula: This calculates the log_softmax for all classes.
    # It computes log_softmax(o_i) = o_i - log(sum(exp(o_i[a]))).
    log_probs = stable_logits - log_sum_exp
    
    # Formula: This selects the specific log probability for the target class x_{i+1}.
    # It effectively performs the indexing: log_softmax(o_i)[x_{i+1}].
    target_log_probs = torch.gather(log_probs, dim=-1, index=x_target.unsqueeze(-1))
    
    # Formula: This applies the negative sign to get the final loss for each example.
    # It calculates the final cross-entropy loss: ℓ_i = -log_softmax(o_i)[x_{i+1}].
    loss_per_example = -target_log_probs.squeeze(-1)
    
    # Formula: This computes the final average loss over the entire batch.
    # It corresponds to: (1 / |D|m) * sum(ℓ_i)
    return torch.mean(loss_per_example)
