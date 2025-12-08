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
    
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    
    // Allocate output for partial sums from each block
    auto output = torch::empty({blocksPerGrid}, input.options());
    
    // Use template for compile-time optimizations
    switch (threadsPerBlock) {
        case 512:
            reduction_kernel<512><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
                input.data_ptr<float>(), output.data_ptr<float>(), N); break;
        case 256:
            reduction_kernel<256><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
                input.data_ptr<float>(), output.data_ptr<float>(), N); break;
        case 128:
            reduction_kernel<128><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
                input.data_ptr<float>(), output.data_ptr<float>(), N); break;
        case 64:
            reduction_kernel<64><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
                input.data_ptr<float>(), output.data_ptr<float>(), N); break;
        default:
            TORCH_CHECK(false, "Unsupported block size");
    }
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    // If we have multiple blocks, recursively reduce until we get one value
    if (blocksPerGrid > 1) {
        return reduction(output);
    }
    
    return output;
}