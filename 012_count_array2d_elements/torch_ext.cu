#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include "main.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor count_equal_kernel_torch(torch::Tensor input, int64_t k) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input tensor must be int32");
    int N = input.numel();
    auto output = torch::zeros({}, input.options().dtype(torch::kInt32)); // Scalar output

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    count_equal_kernel<<<grid, block>>>(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        N,
        static_cast<int>(k)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}