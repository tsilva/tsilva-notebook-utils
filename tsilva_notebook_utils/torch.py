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
