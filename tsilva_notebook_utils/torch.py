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


def calc_model_layer_grad_norms(model):
    """
    Calculates the L2 norm of gradients for each parameter in the model.

    Args:
        model (torch.nn.Module): The model containing parameters with gradients.

    Returns:
        layer_norms (dict): A dictionary mapping parameter names to their individual L2 gradient norms.
    """
    layer_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_norms[name] = param.grad.data.norm(2).item()
    return layer_norms


def calc_model_total_grad_norm(model, layer_norms=None):
    """
    Calculates the total L2 norm of gradients across all parameters in the model,
    using the individual parameter norms.

    Args:
        model (torch.nn.Module): The model containing parameters with gradients.
        layer_norms (dict, optional): A dictionary mapping parameter names to their individual L2 gradient norms.
            If provided, this will be used instead of calculating them again.

    Returns:
        total_norm (float): The total L2 norm of all parameter gradients.
        layer_norms (dict): A dictionary mapping parameter names to their individual L2 gradient norms.
    """
    layer_norms = layer_norms if layer_norms else calc_model_layer_grad_norms(model)
    total_norm_sq = sum(norm ** 2 for norm in layer_norms.values())
    total_norm = total_norm_sq ** 0.5
    return total_norm

def calc_model_grad_norms(model):
    """
    Calculates the L2 norm of gradients for each parameter in the model and the total L2 norm.
    Args:
        model (torch.nn.Module): The model containing parameters with gradients.
    Returns:
        total_norm (float): The total L2 norm of all parameter gradients.
        layer_norms (dict): A dictionary mapping parameter names to their individual L2 gradient norms.
    """
    layer_norms = calc_model_layer_grad_norms(model)
    total_norm = calc_model_total_grad_norm(model, layer_norms)
    return total_norm, layer_norms

def get_model_parameter_counts(model):
    """
    Returns a dictionary with total, trainable, and non-trainable parameter counts in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        dict: Dictionary containing counts of parameters.
            {
                'total': int,
                'trainable': int,
                'non_trainable': int
            }
    """
    total = 0
    trainable = 0
    non_trainable = 0

    for p in model.parameters():
        param_count = p.numel()
        total += param_count
        if p.requires_grad:
            trainable += param_count
        else:
            non_trainable += param_count

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable
    }


def get_device_from_model(model):
    """
    Returns the device of the first parameter in the model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        torch.device: The device of the first parameter.
    """
    for param in model.parameters():
        return param.device
    return None
