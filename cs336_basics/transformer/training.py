import torch
import numpy as np

def data_loading(dataset: np.array, batch_size: int, context_length: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    sampled_input_list = []
    sampled_target_list = []

    # # Sample B random starting indices for your batches.
    start_indices = np.random.randint(low=0, high=(len(dataset) - context_length), size=batch_size)

    for i in start_indices:
        sampled_input_sq = dataset[i: i + context_length]
        sampled_target_sq = dataset[i + 1: i + context_length + 1]

        sampled_input_list.append(torch.from_numpy(sampled_input_sq))
        sampled_target_list.append(torch.from_numpy(sampled_target_sq))
    
    input_tensor = torch.stack(sampled_input_list, dim=0).to(device)
    target_tensor = torch.stack(sampled_target_list, dim=0).to(device)
    return (input_tensor, target_tensor)
        