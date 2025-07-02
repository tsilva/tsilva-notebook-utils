import numpy as np


def to_numpy(data):
    """
    Converts input to a NumPy array. Handles PyTorch tensors.
    """
    
    import torch

    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)

def reshape_vector_to_grid(vector, max_width=16):
    """
    Reshape a 1D NumPy array (or 1D tensor) to 2D grid for visualization.
    Pads with zeros if needed to make it fit the grid.
    """

    vector = to_numpy(vector)

    if vector.ndim == 2 and vector.shape[0] == 1:
        vector = vector.squeeze(0)
    elif vector.ndim != 1:
        raise ValueError("Expected a 1D array or shape (1, D) for grid reshaping.")

    dim = vector.shape[0]
    width = min(dim, max_width)
    height = (dim + width - 1) // width  # Ceiling division
    padded_size = width * height
    padded = np.zeros(padded_size, dtype=vector.dtype)
    padded[:dim] = vector
    return padded.reshape((height, width))
