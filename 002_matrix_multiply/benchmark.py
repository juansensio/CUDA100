import time
import torch
from torch.utils.cpp_extension import load

module = load(
    name='matrix_multiply_naive',
    sources=['naive.cpp', 'naive.cu'],
    extra_cuda_cflags=['-O2'],
    verbose=True
)

def torch_matrix_multiply_naive(A, B):
    return A @ B

a = torch.randn((10, 10), device='cuda')
b = torch.randn((10, 10), device='cuda')

c1 = torch_matrix_multiply_naive(a, b)
c2 = module.matrix_multiply_naive(a, b)
assert torch.allclose(c1, c2, atol=1e-5)

Ns = [512, 1024, 2048, 4096]
rounds = 10

print("\nBenchmarking Custom and PyTorch for various sizes...")
print(f"{'N':<10} {'Custom (ms)':<18} {'GFLOPs':<18} {'PyTorch (ms)':<18} {'GFLOPs':<18} {'Speedup':<18}")

# We'll use only square matrices for benchmarking simplicity
for size in Ns:
    M = N = K = size
    pytorch_times = []
    custom_times = []
    # Prepare random matrices of size (size x size)
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')

    # Warm-up runs
    for _ in range(2):
        _ = torch_matrix_multiply_naive(a, b)
        _ = module.matrix_multiply_naive(a, b)
    torch.cuda.synchronize()

    for _ in range(rounds):
        torch.cuda.synchronize()
        start_time = time.time()
        c1 = torch_matrix_multiply_naive(a, b)
        torch.cuda.synchronize()
        pytorch_times.append(time.time() - start_time)

        torch.cuda.synchronize()
        start_time = time.time()
        c2 = module.matrix_multiply_naive(a, b)
        torch.cuda.synchronize()
        custom_times.append(time.time() - start_time)

    pytorch_mean_ms = (sum(pytorch_times) / rounds) * 1000  # milliseconds
    custom_mean_ms = (sum(custom_times) / rounds) * 1000    # milliseconds

    # Calculate GFLOPs: 2 * M * N * K operations for matrix multiply
    # GFLOPs = (operations) / (time_in_seconds * 1e9)
    # Or: GFLOPs = (2 * M * N * K) / (time_in_ms * 1e6)
    operations = 2.0 * M * N * K
    custom_gflops = operations / (custom_mean_ms * 1e6)
    pytorch_gflops = operations / (pytorch_mean_ms * 1e6)
    speedup_vs_pytorch = custom_gflops / pytorch_gflops  # Speedup of custom vs PyTorch

    print(f"{size:<10} {custom_mean_ms:<18.4f} {custom_gflops:<18.4f} {pytorch_mean_ms:<18.4f} {pytorch_gflops:<18.4f} {speedup_vs_pytorch:<18.4f}")