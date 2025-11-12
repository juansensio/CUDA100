import time
import csv


import torch
from torch.utils.cpp_extension import load

module = load(
    name='vector_add',
    sources=['pytorch_ext.cpp', 'pytorch_ext.cu'],
    extra_cuda_cflags=['-O2'],
    verbose=True
)

def torch_vector_add(a, b):
    return a + b

a = torch.randn(1000, device='cuda')
b = torch.randn(1000, device='cuda')

c1 = torch_vector_add(a, b)
c2 = module.vectorAdd(a, b)
assert torch.allclose(c1, c2) # ok

Ns = [10**i for i in range(6, 10)]  
rounds = 10
results = []
for N in Ns:
    pytorch_times = []
    custom_times = []
    # Prepare random vectors of size N
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    for _ in range(rounds):
        torch.cuda.synchronize()
        start_time = time.time()
        c1 = torch_vector_add(a, b)
        torch.cuda.synchronize()
        pytorch_times.append(time.time() - start_time)

        torch.cuda.synchronize()
        start_time = time.time()
        c2 = module.vectorAdd(a, b)
        torch.cuda.synchronize()
        custom_times.append(time.time() - start_time)
    pytorch_mean = (sum(pytorch_times) / rounds) * 1000  # milliseconds
    custom_mean = (sum(custom_times) / rounds) * 1000    # milliseconds
    print(f"N={N}: PyTorch mean={pytorch_mean:.6f}s, Custom mean={custom_mean:.6f}s")
    results.append({'N': N, 'pytorch': pytorch_mean, 'cuda': custom_mean})
with open('vector_add_benchmark.csv', 'w', newline='') as csvfile:
    fieldnames = ['N', 'pytorch', 'cuda']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)