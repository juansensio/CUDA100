#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include "main.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor copy_matrix_kernel(torch::Tensor input) {
    CHECK_INPUT(input);
    int N = input.size(0);
    auto output = torch::empty({N, N}, input.options());
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    copy_matrix_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}