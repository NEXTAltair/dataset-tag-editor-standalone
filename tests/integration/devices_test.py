import torch

cpu = torch.device("cpu")

def get_cuda_device():
    return torch.device("cuda")

def get_optimal_device():
    if torch.cuda.is_available():
        return get_cuda_device()
    return torch.device("cpu")

device = get_optimal_device()