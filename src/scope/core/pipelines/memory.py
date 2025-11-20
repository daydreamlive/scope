import torch

CUDA_LOW_MEMORY_THRESHOLD = 32  # GB


def get_cuda_free_memory_gb(device: torch.device):
    memory_stats = torch.cuda.memory_stats(device)
    bytes_active = memory_stats["active_bytes.all.current"]
    bytes_reserved = memory_stats["reserved_bytes.all.current"]
    bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
    return bytes_total_available / (1024**3)


def is_cuda_low_memory(device: torch.device):
    return get_cuda_free_memory_gb(device) < CUDA_LOW_MEMORY_THRESHOLD
