import time
import torch
from torch.utils.cpp_extension import load

module = load(
    name='matrix_add',
    sources=['torch_ext.cpp', 'torch_ext.cu'],
    extra_cuda_cflags=['-O2'],
    verbose=True
)

def torch_matrix_add(A, B):
    return A + B

A = torch.randn(2, 2, device='cuda', dtype=torch.float32)
B = torch.randn(2, 2, device='cuda', dtype=torch.float32)

c1 = torch_matrix_add(A, B)
c2 = module.matrix_add(A, B)
assert torch.allclose(c1, c2, atol=1e-5)

Ns = [512, 1024, 2048, 4096]
rounds = 10

print("\nBenchmarking Custom and PyTorch for various sizes...")
print(f"{'N':<10} {'Custom (ms)':<18} {'TFLOPs':<18} {'PyTorch (ms)':<18} {'TFLOPs':<18} {'Speedup':<18}")

# We'll use only square matrices for benchmarking simplicity
for size in Ns:
    N = size
    pytorch_times = []
    custom_times = []
    # Prepare random matrices of size (N x N)
    A = torch.randn(N, N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, N, device='cuda', dtype=torch.float32)
    numel = N * N  # Number of elements for FLOP calculation

    # Warm-up runs
    for _ in range(2):
        c1 = torch_matrix_add(A, B)
        c2 = module.matrix_add(A, B)
        assert torch.allclose(c1, c2, atol=1e-5)
    torch.cuda.synchronize()

    for _ in range(rounds):
        torch.cuda.synchronize()
        start_time = time.time()
        c1 = torch_matrix_add(A, B)
        torch.cuda.synchronize()
        pytorch_times.append(time.time() - start_time)

        torch.cuda.synchronize()
        start_time = time.time()
        c2 = module.matrix_add(A, B)
        torch.cuda.synchronize()
        custom_times.append(time.time() - start_time)

    pytorch_mean_ms = (sum(pytorch_times) / rounds) * 1000  # milliseconds
    custom_mean_ms = (sum(custom_times) / rounds) * 1000    # milliseconds

    # TFLOPs calculation: each add is 1 FLOP, so numel FLOPs total
    # TFLOP/s = numel / (ms * 1e6)
    custom_tflops = numel / (custom_mean_ms * 1e6) if custom_mean_ms > 0 else 0.0
    pytorch_tflops = numel / (pytorch_mean_ms * 1e6) if pytorch_mean_ms > 0 else 0.0

    # Speedup: PyTorch time / Custom time (how much faster PyTorch is)
    speedup_vs_pytorch = pytorch_mean_ms / custom_mean_ms if custom_mean_ms > 0 else 0.0

    print(f"{N:<10} {custom_mean_ms:<18.4f} {custom_tflops:<18.6f} {pytorch_mean_ms:<18.4f} {pytorch_tflops:<18.6f} {speedup_vs_pytorch:<18.4f}")