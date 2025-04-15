def get_current_device():
    import torch
    return torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")


def get_gpu_stats():
    import torch
    assert torch.cuda.is_available(), "CUDA is not available. Running on CPU."

    device = get_current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
    cached_memory = torch.cuda.memory_reserved(device) / 1024**3
    free_memory = total_memory - allocated_memory

    return {
        "total_memory_gb": total_memory,
        "allocated_memory_gb": allocated_memory,
        "cached_memory_gb": cached_memory,
        "free_memory_gb": free_memory
    }


def inspect_tensor_dataset(dataset, show_samples=3):
    """
    Returns a dictionary describing a TensorDataset.
    
    Args:
        dataset (TensorDataset): The dataset to describe.
        show_samples (int): Number of samples to preview (default is 3).
        
    Returns:
        dict: A dictionary with dataset information.
    """
    
    info = {
        "num_samples": len(dataset),
        "tensor_shapes": [tensor.shape for tensor in dataset.tensors],
        "sample_data": []
    }

    for i in range(min(show_samples, len(dataset))):
        sample = dataset[i]
        info["sample_data"].append(tuple(s.clone().detach() for s in sample))  # safely detach from graph

    return info


def apply_weight_init(model, weight_init, nonlinearity):
    """
    Applies weight initialization to the parameters of a given model.

    Parameters:
    ----------
    model : torch.nn.Module
        The PyTorch model whose parameters will be initialized.
    weight_init : str
        The initialization method to use for weights. Supported options are:
        - 'xavier' : Xavier/Glorot uniform initialization
        - 'kaiming' : Kaiming/He uniform initialization
    nonlinearity : str
        The nonlinearity used after the layer (e.g., 'relu', 'tanh'). This is used
        to calculate the gain for initialization.

    Returns:
    -------
    model : torch.nn.Module
        The model with initialized weights and biases.
    """
    import torch.nn as nn

    for name, param in model.named_parameters():
        if 'weight' in name:
            if weight_init == 'xavier':
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain(nonlinearity))
            elif weight_init == 'kaiming':
                nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity=nonlinearity)
        elif 'bias' in name:
            nn.init.zeros_(param)

    return model
