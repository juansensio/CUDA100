#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include "main.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor reduction(torch::Tensor input) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    int N = input.numel();
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    // Allocate output for partial sums from each block
    auto output = torch::empty({(int)grid.x}, input.options());
    
    reduction_kernel<<<grid, block, block.x * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    // If we have multiple blocks, recursively reduce until we get one value
    if (grid.x > 1) {
        return reduction(output);
    }
    
    return output;
}