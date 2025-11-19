#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include "main.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor invert_image(torch::Tensor image) {
    CHECK_INPUT(image);
    int width = image.size(0);
    int height = image.size(1);
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x, 
        (height + block.y - 1) / block.y
    );
    invert_kernel<<<grid, block>>>(image.data_ptr<unsigned char>(), width, height);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return image;
}