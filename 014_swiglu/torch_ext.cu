#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include "main.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor swiglu(torch::Tensor input) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    int N = input.numel();
    int halfN = N / 2;
    auto output = torch::empty({halfN}, input.options());

    dim3 block(256);
    dim3 grid((halfN + block.x - 1) / block.x);

    swiglu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        halfN
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}