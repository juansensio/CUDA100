#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include "main.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor conv1d(torch::Tensor input, torch::Tensor kernel) {
    CHECK_INPUT(input); CHECK_INPUT(kernel);
    int input_size = input.size(0);
    int kernel_size = kernel.size(0);
    int output_size = input_size - kernel_size + 1;
    auto output = torch::empty({output_size}, input.options());
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input.data_ptr<float>(), kernel.data_ptr<float>(), output.data_ptr<float>(), input_size, kernel_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}