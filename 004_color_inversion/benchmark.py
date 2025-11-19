import time
import torch
from torch.utils.cpp_extension import load

module = load(
    name='invert_image',
    sources=['torch_ext.cpp', 'torch_ext.cu'],
    extra_cuda_cflags=['-O2'],
    verbose=True
)

def torch_invert_image(image):
    image[..., :3] = 255 - image[..., :3]
    return image

image = torch.randint(0, 256, (2, 3, 4), device='cuda', dtype=torch.uint8)

c1 = torch_invert_image(image)
c2 = module.invert_image(image)
assert torch.allclose(c1, c2, atol=1e-5)

Ns = [256, 512, 1024, 2048, 4096]
rounds = 10

print("\nBenchmarking Custom and PyTorch for various sizes...")
print(f"{'Width':<10} {'Height':<10} {'Custom (ms)':<18} {'PyTorch (ms)':<18} {'Speedup':<18}")

# We'll use only square matrices for benchmarking simplicity
for size in Ns:
    width = size
    height = size
    pytorch_times = []
    custom_times = []
    # Prepare random matrices of size (size x size)
    image = torch.randint(0, 256, (width, height, 4), device='cuda', dtype=torch.uint8)

    # Warm-up runs
    for _ in range(2):
        _ = torch_invert_image(image)
        _ = module.invert_image(image)
    torch.cuda.synchronize()

    for _ in range(rounds):
        torch.cuda.synchronize()
        start_time = time.time()
        c1 = torch_invert_image(image)
        torch.cuda.synchronize()
        pytorch_times.append(time.time() - start_time)

        torch.cuda.synchronize()
        start_time = time.time()
        c2 = module.invert_image(image)
        torch.cuda.synchronize()
        custom_times.append(time.time() - start_time)

    pytorch_mean_ms = (sum(pytorch_times) / rounds) * 1000  # milliseconds
    custom_mean_ms = (sum(custom_times) / rounds) * 1000    # milliseconds

    # Speedup: PyTorch time / Custom time (how much faster PyTorch is)
    speedup_vs_pytorch = pytorch_mean_ms / custom_mean_ms  # >1 means PyTorch is faster

    print(f"{width:<10} {height:<10} {custom_mean_ms:<18.4f} {pytorch_mean_ms:<18.4f} {speedup_vs_pytorch:<18.4f}")