#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include "main.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor relu_kernel(torch::Tensor input) {
    CHECK_INPUT(input);
    int N = input.size(0);
    auto output = torch::empty({N}, input.options());
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}