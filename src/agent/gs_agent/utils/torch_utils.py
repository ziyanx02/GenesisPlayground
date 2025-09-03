import torch
import torch.nn as nn


def get_activation(activation: str | type[nn.Module] | nn.Module) -> nn.Module:
    """Return an activation module instance given a name, class, or instance."""
    # If the user passed in an nn.Module instance, just return it
    if isinstance(activation, nn.Module):
        return activation

    # If they passed in a class (e.g. nn.ReLU), instantiate it
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation()

    # Otherwise assume it's a string
    name = activation.lower().strip()  # type: ignore[union-attr]
    activations: dict[str, nn.Module] = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "softmax": nn.Softmax(dim=-1),
        "none": nn.Identity(),
    }
    if name not in activations:
        raise ValueError(
            f"Unsupported activation: {activation!r}. "
            f"Supported options: {list(activations.keys()) + ['nn.Module subclass', 'nn.Module instance']}"
        )
    return activations[name]


def get_torch_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA is not available")
    else:
        print("CUDA is available")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    return device
