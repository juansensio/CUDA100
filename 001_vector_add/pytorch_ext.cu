#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

__global__ void vectorAddKernel(float *out, float *a, float *b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

torch::Tensor vectorAdd(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a); CHECK_INPUT(b);
    int n = a.size(0);
    auto output = torch::empty(n, a.options());
    int threads = 256;
    vectorAddKernel<<<cdiv(n, threads), threads>>>(output.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}