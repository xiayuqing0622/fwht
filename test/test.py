# Copyright (C) 2023, Tri Dao.

import math

import torch
import torch.nn.functional as F
import pytest

from einops import rearrange, repeat
import time
# from fast_hadamard_transform.fast_hadamard_transform_interface import hadamard_transform, hadamard_transform_ref
from fwht import fast_hadamard_transform as fwht_fast_hadamard_transform
from fast_hadamard_transform import hadamard_transform # pip install fast_hadamard_transform
try:
    from scipy.linalg import hadamard
except ImportError:
    hadamard = None
def hadamard_transform_ref(x, scale=1.0):
    """
    x: (..., dim)
    out: (..., dim)
    """
    if hadamard is None:
        raise ImportError("Please install scipy")
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2 ** log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device))
    out = out * scale
    return out[..., :dim].reshape(*x_shape)

# # @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# # @pytest.mark.parametrize("dim", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 137, 1024, 2048, 4096, 8192, 16384, 32768])
# # @pytest.mark.parametrize("dim", [16, 32, 64, 128, 256, 512, 137, 1024, 2048, 4096, 8192])#, 16384, 32768])
# @pytest.mark.parametrize("dim", [4096])
def test_fast_hadamard_transform(dim, dtype):
    device = "cuda"
    rtol, atol = (3e-4, 3e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    torch.random.manual_seed(0)
    batch_size = 15
    # batch_size = 1
    x = torch.randn(batch_size, dim, device=device, dtype=dtype).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    x_pt = x.detach().clone().requires_grad_()
    scale = 1 / math.sqrt(dim)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        out = hadamard_transform(x, scale=scale)
        # out = fwht_fast_hadamard_transform(x, scale=scale)
    torch.cuda.synchronize()
    print(f"Time: {(time.time() - start):.4f} ms")
    out_ref = hadamard_transform_ref(x_ref.float(), scale=scale)
    out_pt = hadamard_transform_ref(x_pt, scale=scale)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Output Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Output Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    assert (out - out_ref).abs().max().item() < 2 * (out_pt - out_ref).abs().max() + atol

    g = torch.randn_like(out)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        out.backward(g)
    torch.cuda.synchronize()
    print(f"Backward Time: {(time.time() - start):.4f} ms")
    # out.backward(g)
    out_ref.backward(g)
    out_pt.backward(g)

    print(f"dx max diff: {(x.grad - x_ref.grad).abs().max().item()}")
    print(f"dx Pytorch max diff: {(x_pt.grad - x_ref.grad).abs().max().item()}")
    assert (x.grad - x_ref.grad).abs().max().item() < 2 * (x_pt.grad - x_ref.grad).abs().max() + atol
test_fast_hadamard_transform(4096, torch.float16)