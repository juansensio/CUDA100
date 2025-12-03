#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include "main.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor silu(torch::Tensor input) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    int N = input.numel();
    auto output = torch::empty({N}, input.options());

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    silu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}