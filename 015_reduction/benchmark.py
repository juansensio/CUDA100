import time
import torch
from torch.utils.cpp_extension import load

module = load(
    name='reduction',
    sources=['torch_ext.cpp', 'torch_ext.cu'],
    extra_cuda_cflags=['-O2'],
    verbose=True
)

def torch_reduction(input):
    return input.sum()

input = (torch.rand(100, device='cuda', dtype=torch.float32) * 2000.0 - 1000.0)
c1 = torch_reduction(input)
c2 = module.reduction(input)
assert torch.allclose(c1, c2, atol=1e-5)

print("\nBenchmarking Custom and PyTorch for various sizes...")
print(f"{'N':<10} {'Custom (ms)':<18} {'Custom (GB/s)':<18} {'PyTorch (ms)':<18} {'PyTorch (GB/s)':<18} {'Speedup':<10}")

sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
num_runs = 10

for N in sizes:
    # Allocate input in [-100, 100] as in C++ code
    input = (torch.rand(N, device='cuda', dtype=torch.float32) * 2000.0 - 1000.0)

    # Warm-up
    module.reduction(input)
    torch.cuda.synchronize()
    torch_reduction(input)
    torch.cuda.synchronize()

    # Validate correctness at each size before benchmarking
    result_custom = module.reduction(input)
    result_torch = torch_reduction(input)
    assert torch.allclose(result_custom, result_torch, atol=1e-5), f"Mismatch at size {N}: custom={result_custom}, torch={result_torch}"

    # Measure custom reduction
    times_custom = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        out = module.reduction(input)
        torch.cuda.synchronize()
        t1 = time.time()
        times_custom.append(t1 - t0)
    ms_custom = 1000 * sum(times_custom) / num_runs

    # Measure PyTorch reduction
    times_torch = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        out = torch_reduction(input)
        torch.cuda.synchronize()
        t1 = time.time()
        times_torch.append(t1 - t0)
    ms_torch = 1000 * sum(times_torch) / num_runs

    # Bandwidth (read + write)
    total_bytes = N * 4  # float32 bytes moved
    custom_gbps = total_bytes / (ms_custom * 1e6)
    pytorch_gbps = total_bytes / (ms_torch * 1e6)

    # Avoid div0; show speedup as PyTorch/custom (how many times faster is custom)
    speedup = ms_torch / ms_custom if ms_custom > 0 else float('inf')

    print(f"{N:<10} {ms_custom:<18.4f} {custom_gbps:<18.2f} {ms_torch:<18.4f} {pytorch_gbps:<18.2f} {speedup:<10.2f}")
