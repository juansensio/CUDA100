import time
import torch
from torch.utils.cpp_extension import load

module = load(
    name='copy_matrix_kernel',
    sources=['torch_ext.cpp', 'torch_ext.cu'],
    extra_cuda_cflags=['-O2'],
    verbose=True
)

def torch_copy_matrix_kernel(input):
    return input.clone()

input = torch.randn(2, 2, device='cuda', dtype=torch.float32)
c1 = torch_copy_matrix_kernel(input)
c2 = module.copy_matrix_kernel(input)
assert torch.allclose(c1, c2, atol=1e-5)

Ns = [256, 512, 1024, 2048, 4096] 
rounds = 10

print("\nBenchmarking Custom and PyTorch for various sizes...")
print(f"{'N':<10} {'Custom (ms)':<18} {'Custom (GB/s)':<18} {'PyTorch (ms)':<18} {'PyTorch (GB/s)':<18} {'Speedup':<10}")

for size in Ns:
    pytorch_times = []
    custom_times = []
    input = torch.randn(size, size, device='cuda', dtype=torch.float32)

    # Warm-up runs
    for _ in range(2):
        c1 = torch_copy_matrix_kernel(input)
        c2 = module.copy_matrix_kernel(input)
        assert torch.allclose(c1, c2, atol=1e-5)
    torch.cuda.synchronize()

    for _ in range(rounds):
        torch.cuda.synchronize()
        start_time = time.time()
        c1 = torch_copy_matrix_kernel(input)
        torch.cuda.synchronize()
        pytorch_times.append(time.time() - start_time)

        torch.cuda.synchronize()
        start_time = time.time()
        c2 = module.copy_matrix_kernel(input)
        torch.cuda.synchronize()
        custom_times.append(time.time() - start_time)

    pytorch_mean_ms = (sum(pytorch_times) / rounds) * 1000  # milliseconds
    custom_mean_ms = (sum(custom_times) / rounds) * 1000    # milliseconds

    num_bytes = size * size * 4 * 2  # float32, read+write, bytes moved
    custom_gbps = num_bytes / (custom_mean_ms * 1e6)
    pytorch_gbps = num_bytes / (pytorch_mean_ms * 1e6)
    # Avoid div0; show speedup as PyTorch/custom (how many times faster is custom)
    speedup = pytorch_mean_ms / custom_mean_ms if custom_mean_ms > 0 else float('inf')

    print(f"{size}x{size:<10} {custom_mean_ms:<18.4f} {custom_gbps:<18.2f} {pytorch_mean_ms:<18.4f} {pytorch_gbps:<18.2f} {speedup:<10.2f}")
