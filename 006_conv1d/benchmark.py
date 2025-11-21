import time
import torch
from torch.utils.cpp_extension import load

module = load(
    name='conv1d',
    sources=['torch_ext.cpp', 'torch_ext.cu'],
    extra_cuda_cflags=['-O2'],
    verbose=True
)

def torch_conv1d(input, kernel):
    # input: (batch, 1, signal_len)  kernel: (kernel_size)
    # PyTorch expects weight of shape (out_channels, in_channels, kernel_size)
    # We'll use groups=batch to map input[b] with kernel, replicating kernel through batch for fair comparison
    # But for now, just do single batch, single channel (like the CUDA kernel)
    # So input: (B, 1, L)  kernel: (K)  weight for PyTorch: (1, 1, K)
    weight = kernel.view(1, 1, -1)
    return torch.nn.functional.conv1d(input, weight, padding=0, stride=1)

# To match the CUDA kernel (and avoid dimension errors), use batch=1 throughout, and input shape (1, 1, length), kernel shape (kernel_size,)
input = torch.randn(1, 1, 100, device='cuda', dtype=torch.float32)
kernel = torch.randn(3, device='cuda', dtype=torch.float32)

c1 = torch_conv1d(input, kernel)
c2 = module.conv1d(input.squeeze(0).squeeze(0), kernel)
# module.conv1d expects 1D input and 1D kernel, returns (output_len)
# For comparison, squeeze the batch/channel dims from pytorch output, or reshape as needed:
assert torch.allclose(c1.squeeze(0).squeeze(0), c2, atol=1e-5)

Ns = [10_000, 100_000, 1_500_000]
kernel_sizes = [512, 1024, 2048]
rounds = 10

print("\nBenchmarking Custom and PyTorch for various sizes and kernel sizes...")
print(f"{'N':<10} {'K':<8} {'Custom (ms)':<18} {'TFLOPs':<18} {'PyTorch (ms)':<18} {'TFLOPs':<18} {'Speedup':<18}")

# Compute FLOPs for fair comparison: conv1d for each output element is kernel_size multiplications and (kernel_size - 1) adds (so, 2*kernel_size - 1 FLOPs)
for size in Ns:
    for kernel_size in kernel_sizes:
        if kernel_size > size:
            # invalid, skip this configuration
            continue

        pytorch_times = []
        custom_times = []
        # Use batch=1, in_channels=1, signal_len
        input = torch.randn(1, 1, size, device='cuda', dtype=torch.float32)
        kernel = torch.randn(kernel_size, device='cuda', dtype=torch.float32)
        out_len = size - kernel_size + 1
        batch = 1
        flops_per_output = 2 * kernel_size - 1
        total_flops = batch * out_len * flops_per_output

        # Warm-up runs
        for _ in range(2):
            c1 = torch_conv1d(input, kernel)
            c2 = module.conv1d(input.squeeze(0).squeeze(0), kernel)
            assert torch.allclose(c1.squeeze(0).squeeze(0), c2, atol=1e-5)
        torch.cuda.synchronize()

        for _ in range(rounds):
            torch.cuda.synchronize()
            start_time = time.time()
            c1 = torch_conv1d(input, kernel)
            torch.cuda.synchronize()
            pytorch_times.append(time.time() - start_time)

            torch.cuda.synchronize()
            start_time = time.time()
            c2 = module.conv1d(input.squeeze(0).squeeze(0), kernel)
            torch.cuda.synchronize()
            custom_times.append(time.time() - start_time)

        pytorch_mean_ms = (sum(pytorch_times) / rounds) * 1000  # milliseconds
        custom_mean_ms = (sum(custom_times) / rounds) * 1000    # milliseconds

        # TFLOPs calculation: TFLOPs = total_flops / (time_in_seconds * 1e12)
        # time_in_ms * 1e-3 = time_in_seconds, so: TFLOPs = total_flops / (time_in_ms * 1e-3 * 1e12) = total_flops / (time_in_ms * 1e9)
        custom_tflops = total_flops / (custom_mean_ms * 1e9) if custom_mean_ms > 0 else 0.0
        pytorch_tflops = total_flops / (pytorch_mean_ms * 1e9) if pytorch_mean_ms > 0 else 0.0

        # Speedup: PyTorch time / Custom time (how much faster PyTorch is)
        speedup_vs_pytorch = pytorch_mean_ms / custom_mean_ms if custom_mean_ms > 0 else 0.0

        print(f"{size:<10} {kernel_size:<8} {custom_mean_ms:<18.4f} {custom_tflops:<18.6f} {pytorch_mean_ms:<18.4f} {pytorch_tflops:<18.6f} {speedup_vs_pytorch:<18.4f}")
