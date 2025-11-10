#include "kernels.h"

__global__ void matrix_multiply_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C, 
    int M, int N, int K
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y* blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float value = 0.0f;
        for (int i = 0; i < K; ++i) {
            value += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

