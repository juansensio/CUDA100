#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include "main.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor reverse_array(torch::Tensor input) {
    CHECK_INPUT(input);
    int N = input.size(0);
    int threadsPerBlock = 256;
    int blocksPerGrid = ((N / 2) + threadsPerBlock - 1) / threadsPerBlock;
    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input.data_ptr<float>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return input;
}