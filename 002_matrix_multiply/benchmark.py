import time
import sys
import torch
from torch.utils.cpp_extension import load
from typing import Callable, List, Tuple

# Load the CUDA extension module
module = load(
    name='matrix_multiply',
    sources=['torch_ext.cpp', 'torch_ext.cu'],
    extra_cuda_cflags=['-O2'],
    verbose=True
)

# Kernel descriptor structure
class KernelDescriptor:
    def __init__(self, name: str, func: Callable, is_baseline: bool = False):
        self.name = name
        self.func = func
        self.is_baseline = is_baseline

# Benchmark result structure
class BenchmarkResult:
    def __init__(self, ms: float, gflops: float, speedup: float = 1.0):
        self.ms = ms
        self.gflops = gflops
        self.speedup = speedup

# Kernel registry - easy to add new kernels here
kernels = [
    KernelDescriptor("PyTorch", lambda A, B: A @ B, is_baseline=True),
    KernelDescriptor("Naive", module.matrix_multiply_naive),
    KernelDescriptor("Coalescing", module.matrix_multiply_coalescing),
    # KernelDescriptor("cuBLAS", module.matrix_multiply_cublas),  # Uncomment when ready
]

def validate_kernel(kernel: KernelDescriptor, A: torch.Tensor, B: torch.Tensor, 
                    expected: torch.Tensor, atol: float = 1e-5) -> bool:
    """Validate that a kernel produces correct results."""
    try:
        result = kernel.func(A, B)
        if torch.allclose(result, expected, atol=atol):
            return True
        else:
            print(f"✗ {kernel.name} kernel validation failed: results don't match!")
            return False
    except Exception as e:
        print(f"✗ {kernel.name} kernel validation failed: {e}")
        return False

def benchmark_kernel(kernel: KernelDescriptor, A: torch.Tensor, B: torch.Tensor,
                     M: int, N: int, K: int, num_runs: int) -> BenchmarkResult:
    """Benchmark a single kernel and return timing results."""
    times = []
    
    # Warm-up runs
    for _ in range(2):
        _ = kernel.func(A, B)
    torch.cuda.synchronize()
    
    # Benchmark runs
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start_time = time.time()
        _ = kernel.func(A, B)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)
    
    mean_ms = (sum(times) / num_runs) * 1000  # Convert to milliseconds
    operations = 2.0 * M * N * K
    gflops = operations / (mean_ms * 1e6)
    
    return BenchmarkResult(mean_ms, gflops, 1.0)

def validate_all_kernels():
    """Validate all kernels with a small test case."""
    print("Testing correctness with small matrices...")
    A = torch.randn((10, 10), device='cuda')
    B = torch.randn((10, 10), device='cuda')
    
    # Use PyTorch as reference
    baseline = kernels[0].func(A, B)
    
    # Validate all kernels (skip baseline)
    for kernel in kernels[1:]:
        if not validate_kernel(kernel, A, B, baseline):
            return False
        print(f"✓ {kernel.name} kernel result is correct!")
    
    return True

def benchmark():
    """Run benchmarks for all kernels across various sizes."""
    print("\nBenchmarking kernels for various sizes...")
    print(f"{'Kernel':<15} {'Size':<12} {'Time (ms)':<12} {'TFLOPs':<12} {'Speedup':<12}")
    print("=" * 63)
    
    sizes = [512, 1024, 2048, 4096]
    num_runs = 10
    
    for size in sizes:
        M = N = K = size
        
        # Prepare matrices
        A = torch.randn(size, size, device='cuda')
        B = torch.randn(size, size, device='cuda')
        
        baseline_ms = 0.0
        
        # Benchmark each kernel and print results immediately
        for kernel in kernels:
            result = benchmark_kernel(kernel, A, B, M, N, K, num_runs)
            
            # First kernel (PyTorch) becomes the baseline
            if kernel.is_baseline:
                baseline_ms = result.ms
            
            # Calculate speedup: baseline_ms / result.ms (fixed: baseline/measured)
            speedup = baseline_ms / result.ms if baseline_ms > 0.0 and not kernel.is_baseline else 1.0
            
            # Print result immediately
            speedup_str = f"{speedup:.2f}" if not kernel.is_baseline else "-"
            print(f"{kernel.name:<15} {size:<12} {result.ms:<12.4f} {result.gflops / 1000.0:<12.4f} {speedup_str:<12}")
            # Force output to appear immediately
            sys.stdout.flush()
        
        print()  # Blank line between sizes

if __name__ == "__main__":
    # Validate kernels first
    if not validate_all_kernels():
        exit(1)
    
    # Run benchmarks
    benchmark()
