import random
from collections import OrderedDict


def get_default_device():
    import torch
    if torch.backends.mps.is_available() and torch.backends.mps.is_built(): return torch.device("mps")
    elif torch.cuda.is_available(): return torch.device("cuda")
    else: return torch.device("cpu")

# TODO: deprecate this function in favor of get_default_device()
def get_current_device():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

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
    import torch
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
    import torch
    import torch.nn as nn
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
    

    for name, param in model.named_parameters():
        if 'weight' in name:
            if weight_init == 'xavier':
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain(nonlinearity))
            elif weight_init == 'kaiming':
                nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity=nonlinearity)
        elif 'bias' in name:
            nn.init.zeros_(param)

    return model


def get_model_grad_norms(model, per_layer=False):
    import torch
    """
    Calculates the L2 norm of gradients for each parameter in the model and/or the total L2 norm.
    Args:
        model (torch.nn.Module): The model containing parameters with gradients.
        per_layer (bool): If True, returns a dict of per-layer gradient norms. If False, returns only the total norm.
    Returns:
        If per_layer is True: (total_norm, layer_norms)
        If per_layer is False: total_norm
    """
    layer_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_norms[name] = param.grad.data.norm(2).item()
    total_norm_sq = sum(norm ** 2 for norm in layer_norms.values())
    total_norm = total_norm_sq ** 0.5
    if per_layer:
        return total_norm, layer_norms
    return total_norm

def get_model_parameter_counts(model):
    """
    Returns a dictionary with total, trainable, and non-trainable parameter counts in a PyTorch model.
    For richer model summaries, consider using torchinfo (https://github.com/TylerYep/torchinfo).
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


def get_module_device(module):
    return next(module.parameters(), next(module.buffers(), None)).device

def get_conv_filter_images(model, nrow=8, padding=1, scale=4):
    import torch
    import torchvision.utils as vutils
    from PIL import Image
    from torchvision.transforms.functional import to_pil_image
    """
    Returns a dict of PIL images visualizing filters in Conv2d and ConvTranspose2d layers.
    
    Args:
        model (torch.nn.Module): The PyTorch model.
        nrow (int): Number of filters per row in the grid.
        padding (int): Padding between filters in the grid.
        scale (int): Factor to scale up the output images.
        
    Returns:
        dict: {layer_name: PIL.Image}
    """


    filter_images = OrderedDict()
    
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            weights = layer.weight.data.clone().cpu()

            # Normalize to [0, 1]
            weights_min = weights.min()
            weights_max = weights.max()
            weights = (weights - weights_min) / (weights_max - weights_min + 1e-5)

            # If it's an RGB-like input (3 channels), show all channels
            if weights.shape[1] == 3:
                grid = vutils.make_grid(weights, nrow=nrow, padding=padding)
            else:
                # Visualize only the first input channel
                grid = vutils.make_grid(weights[:, 0:1, :, :], nrow=nrow, padding=padding)
            
            # Convert to PIL
            pil_img = to_pil_image(grid)

            # Resize for visibility
            if scale > 1:
                new_size = (pil_img.width * scale, pil_img.height * scale)
                pil_img = pil_img.resize(new_size, resample=Image.NEAREST)

            filter_images[name] = pil_img
            
    return filter_images


def configure_matmul_precision():
    import torch
    if torch.cuda.is_available():
        # Get compute capability of the current CUDA device
        major, minor = torch.cuda.get_device_capability()
        # TF32 is available on Ampere (compute capability >= 8.0)
        if major >= 8:
            torch.set_float32_matmul_precision('high')
            print("TF32 matmul precision set to 'high'")
        else:
            print("TF32 not supported on this GPU, skipping matmul precision setting")
    else:
        print("CUDA not available, matmul precision setting skipped")


def create_infinite_data_loader():
    import torch
    class InfiniteDataLoader(torch.utils.data.IterableDataset):
        def __iter__(self):
            while True: yield torch.tensor(0) 
    dataloader = torch.utils.data.DataLoader(InfiniteDataLoader(), batch_size=1, num_workers=2, shuffle=False)
    return dataloader


def seed_everything(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
