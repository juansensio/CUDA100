import time
import torch
from torch.utils.cpp_extension import load

module = load(
    name='matrix_transpose',
    sources=['torch_ext.cpp', 'torch_ext.cu'],
    extra_cuda_cflags=['-O2'],
    verbose=True
)

def torch_matrix_transpose(A):
    return A.T # creo que esto no mueve datos, solo cambia el view del tensor

a = torch.randn((10, 10), device='cuda')

c1 = torch_matrix_transpose(a)
c2 = module.matrix_transpose(a)
assert torch.allclose(c1, c2, atol=1e-5)

Ns = [1024, 2048, 4096, 8192]
rounds = 10

print("\nBenchmarking Custom and PyTorch for various sizes...")
print(f"{'N':<10} {'Custom (ms)':<18} {'PyTorch (ms)':<18} {'Speedup':<18}")

# We'll use only square matrices for benchmarking simplicity
for size in Ns:
    N = size
    M = size
    pytorch_times = []
    custom_times = []
    # Prepare random matrices of size (size x size)
    a = torch.randn(size, size, device='cuda')

    # Warm-up runs
    for _ in range(2):
        _ = torch_matrix_transpose(a)
        _ = module.matrix_transpose(a)
    torch.cuda.synchronize()

    for _ in range(rounds):
        torch.cuda.synchronize()
        start_time = time.time()
        c1 = torch_matrix_transpose(a)
        torch.cuda.synchronize()
        pytorch_times.append(time.time() - start_time)

        torch.cuda.synchronize()
        start_time = time.time()
        c2 = module.matrix_transpose(a)
        torch.cuda.synchronize()
        custom_times.append(time.time() - start_time)

    pytorch_mean_ms = (sum(pytorch_times) / rounds) * 1000  # milliseconds
    custom_mean_ms = (sum(custom_times) / rounds) * 1000    # milliseconds

    # Speedup: PyTorch time / Custom time (how much faster PyTorch is)
    speedup_vs_pytorch = pytorch_mean_ms / custom_mean_ms  # >1 means PyTorch is faster

    print(f"{size:<10} {custom_mean_ms:<18.4f} {pytorch_mean_ms:<18.4f} {speedup_vs_pytorch:<18.4f}")