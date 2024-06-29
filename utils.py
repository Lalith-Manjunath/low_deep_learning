import torch
import inspect


def tensor_info(tensor, custom_name=None):
    """Prints debug information about a PyTorch tensor, including its variable name.

    Args:
        tensor: The PyTorch tensor to inspect.
        custom_name (optional): A custom name to override the automatic variable name detection.
    """
    for frame_info in inspect.stack():
        for var_name, obj in frame_info.frame.f_locals.items():
            if obj is tensor:
                tensor_name = var_name if custom_name is None else custom_name
                break

    print(f"\n--- {tensor_name} Info ---")
    print(f"Shape: {tensor.shape}")
    print(f"Data Type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    # print(f"Values:\n{tensor}")
