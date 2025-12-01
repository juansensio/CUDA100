import time
import torch
from torch.utils.cpp_extension import load

module = load(
    name='count_equal_kernel',
    sources=['torch_ext.cpp', 'torch_ext.cu'],
    extra_cuda_cflags=['-O2'],
    verbose=True
)

def torch_count_equal_kernel(input, k):
    # Ensure the output is int32 to match the custom kernel for comparison
    return torch.sum(input == k).to(torch.int32)

input = torch.randint(0, 100000, (1000000,), device='cuda', dtype=torch.int32)
c1 = torch_count_equal_kernel(input, 1)
c2 = module.count_equal_kernel_torch(input, 1)
assert torch.allclose(c1, c2, atol=1e-5)

Ns = [1000000, 10000000, 100000000, 1000000000] 
Ks = [1, 50000, 100000]
rounds = 10

print("\nBenchmarking Custom and PyTorch for various sizes...")
print(f"{'N':<10} {'K':<6} {'Custom (ms)':<18} {'Custom (GB/s)':<18} {'PyTorch (ms)':<18} {'PyTorch (GB/s)':<18} {'Speedup':<10}")

for k in Ks:
    for size in Ns:
        pytorch_times = []
        custom_times = []
        input = torch.randint(0, 100000, (size,), device='cuda', dtype=torch.int32)

        # Warm-up runs
        for _ in range(2):
            c1 = torch_count_equal_kernel(input, k)
            c2 = module.count_equal_kernel_torch(input, k)
            assert torch.allclose(c1, c2, atol=1e-5)
        torch.cuda.synchronize()

        for _ in range(rounds):
            torch.cuda.synchronize()
            start_time = time.time()
            c1 = torch_count_equal_kernel(input, k)
            torch.cuda.synchronize()
            pytorch_times.append(time.time() - start_time)

            torch.cuda.synchronize()
            start_time = time.time()
            c2 = module.count_equal_kernel_torch(input, k)
            torch.cuda.synchronize()
            custom_times.append(time.time() - start_time)

        pytorch_mean_ms = (sum(pytorch_times) / rounds) * 1000  # milliseconds
        custom_mean_ms = (sum(custom_times) / rounds) * 1000    # milliseconds

        num_bytes = size * 4  # int32, bytes moved
        custom_gbps = num_bytes / (custom_mean_ms * 1e6)
        pytorch_gbps = num_bytes / (pytorch_mean_ms * 1e6)
        # Avoid div0; show speedup as PyTorch/custom (how many times faster is custom)
        speedup = pytorch_mean_ms / custom_mean_ms if custom_mean_ms > 0 else float('inf')

        print(f"{size:<10} {k:<6} {custom_mean_ms:<18.4f} {custom_gbps:<18.2f} {pytorch_mean_ms:<18.4f} {pytorch_gbps:<18.2f} {speedup:<10.2f}")
