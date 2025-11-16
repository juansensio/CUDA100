#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include "main.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor matrix_transpose(torch::Tensor A) {
    CHECK_INPUT(A);
    int n = A.size(0);
    int m = A.size(1);
    auto output = torch::empty({m, n}, A.options());
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((m + TILE_DIM - 1) / TILE_DIM,
              (n + TILE_DIM - 1) / TILE_DIM);
    matrix_transpose<<<grid, block>>>(
        A.data_ptr<float>(),
        output.data_ptr<float>(),
        n, m
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}