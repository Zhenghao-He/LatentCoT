import torch
def add_with_zero_pad(a, b):
    L = max(a.numel(), b.numel())
    out = torch.zeros(L, device=a.device, dtype=a.dtype)
    out[:a.numel()] += a
    out[:b.numel()] += b
    return out