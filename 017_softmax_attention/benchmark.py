import time
import torch
from torch.utils.cpp_extension import load
import math

module = load(
    name='softmax_attention',
    sources=['torch_ext.cpp', 'torch_ext.cu'],
    extra_cuda_cflags=['-O2'],
    verbose=True
)

def torch_softmax_attention(Q, K, V):
    return torch.nn.functional.softmax(Q @ K.T / math.sqrt(Q.size(1)), dim=1) @ V

Q = torch.rand(2, 4, device='cuda', dtype=torch.float32)
K = torch.rand(3, 4, device='cuda', dtype=torch.float32)
V = torch.rand(3, 4, device='cuda', dtype=torch.float32)
c1 = torch_softmax_attention(Q, K, V)
c2 = module.softmax_attention(Q, K, V)
c3 = module.softmax_attention_cublas(Q, K, V)
assert torch.allclose(c1, c2, atol=1e-3)
assert torch.allclose(c1, c3, atol=1e-3)

print("\nBenchmarking Custom and PyTorch Softmax Attention (Q: Mxd, K/V: Nxd) for various sizes...")
print(f"{'M':<7} {'N':<7} {'d':<7} {'Custom (ms)':<15} {'Custom (TFLOPs)':<18} {'PyTorch (ms)':<15} {'PyTorch (TFLOPs)':<18} {'Speedup':<10}")

# Test grid: (M, N, d)
benchmark_configs = [
    (128, 128, 64),
    (1024, 1024, 128),
    (4096, 4096, 256),
    (8192, 8192, 512),
    (16_384, 16_384, 1024),
]
num_runs = 10

for M, N, d in benchmark_configs:
    Q = torch.rand(M, d, device='cuda', dtype=torch.float32)
    K = torch.rand(N, d, device='cuda', dtype=torch.float32)
    V = torch.rand(N, d, device='cuda', dtype=torch.float32)

    # Warm-up
    module.softmax_attention_cublas(Q, K, V)
    torch.cuda.synchronize()
    torch_softmax_attention(Q, K, V)
    torch.cuda.synchronize()

    # Validate correctness
    result_custom = module.softmax_attention_cublas(Q, K, V)
    result_torch = torch_softmax_attention(Q, K, V)
    assert torch.allclose(result_custom, result_torch, atol=1e-1), f"Mismatch for (M={M},N={N},d={d})"

    # Measure custom kernel
    times_custom = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = module.softmax_attention_cublas(Q, K, V)
        torch.cuda.synchronize()
        times_custom.append(time.time() - t0)
    ms_custom = 1000 * sum(times_custom) / num_runs

    # Measure PyTorch
    times_torch = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = torch_softmax_attention(Q, K, V)
        torch.cuda.synchronize()
        times_torch.append(time.time() - t0)
    ms_torch = 1000 * sum(times_torch) / num_runs

    # Estimate FLOPs for attention (see Appendix in https://arxiv.org/pdf/2009.06732.pdf)
    # Q@K^T: 2*M*N*d (matmul, Mxd @ d x N)
    # Softmax: ~2*M*N (exp & norm, roughly)
    # Softmax@V: 2*M*N*d (M x N @ N x d)
    # Total: 4*M*N*d + 2*M*N (but for large d, 4*M*N*d dominates)
    flops = 4 * M * N * d
    tflops_custom = flops / (ms_custom * 1e-3) / 1e12
    tflops_torch = flops / (ms_torch * 1e-3) / 1e12

    speedup = ms_torch / ms_custom if ms_custom > 0 else float('inf')

    print(f"{M:<7} {N:<7} {d:<7} {ms_custom:<15.4f} {tflops_custom:<18.2f} {ms_torch:<15.4f} {tflops_torch:<18.2f} {speedup:<10.2f}")
