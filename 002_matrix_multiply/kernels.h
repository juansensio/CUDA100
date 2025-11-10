#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_OK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

inline unsigned int CEIL_DIV(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

#define BLOCKSIZE 32

// Kernel declarations
__global__ void matrix_multiply_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C, 
    int M, int N, int K
);

__global__ void matrix_multiply_coalescing(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C, 
    int M, int N, int K
);

#endif // KERNELS_H

