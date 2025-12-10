#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>

#include "matrix_transpose.cu"
#include "matrix_multiply.cu"
#include "matrix_divide.cu"
#include "softmax.cu"

#define CUDA_OK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

inline unsigned int CEIL_DIV(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

// my version using previous kernels (slow)
void softmax_attention(
    const float* Q, 
    const float* K, 
    const float* V, 
    float* output, 
    int M, int N, int d) 
{
    // transpose K
    float *K_d, *Kt_d; 
    CUDA_OK(cudaMalloc((void**)&K_d, N * d * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&Kt_d, d * N * sizeof(float)));
    CUDA_OK(cudaMemcpy(K_d, K, N * d * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((d + TILE_DIM - 1) / TILE_DIM,
              (N + TILE_DIM - 1) / TILE_DIM);
    matrix_transpose<<<grid, block>>>(K_d, Kt_d, N, d);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaFree(K_d));

    // multiply Q and Kt
    float *Q_d, *QKt_d;
    CUDA_OK(cudaMalloc((void**)&Q_d, M * d * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&QKt_d, M * N * sizeof(float)));
    CUDA_OK(cudaMemcpy(Q_d, Q, M * d * sizeof(float), cudaMemcpyHostToDevice));
    block = dim3((BM * BN) / (TM * TN));
    grid = dim3(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    matrix_multiply_tiling<<<grid, block>>>(Q_d, Kt_d, QKt_d, M, N, d);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaFree(Q_d));
    CUDA_OK(cudaFree(Kt_d));

    // // divide QKt by sqrt(d)
    // block = dim3(256);
    // grid = dim3(CEIL_DIV(M * N, block.x));
    // matrix_divide<<<grid, block>>>(QKt_d, sqrtf(d), M, N);

    // // row-wise softmax
    // for (int i = 0; i < M; ++i) {
    //     softmax(QKt_d + i * N, QKt_d + i * N, N);
    // }

    // batched divide softmax
    block = dim3(256);
    grid = dim3(M);
    batched_divide_softmax_kernel<<<grid, block>>>(QKt_d, sqrtf(d), M, N);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());

    // multiply softmax(QKt/sqrt(d)) and V
    float *V_d, *output_d;
    CUDA_OK(cudaMalloc((void**)&V_d, N * d * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&output_d, M * d * sizeof(float)));
    CUDA_OK(cudaMemcpy(V_d, V, N * d * sizeof(float), cudaMemcpyHostToDevice));
    block = dim3((BM * BN) / (TM * TN));
    grid = dim3(CEIL_DIV(d, BN), CEIL_DIV(M, BM));
    matrix_multiply_tiling<<<grid, block>>>(QKt_d, V_d, output_d, M, d, N);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(output, output_d, M * d * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(V_d));
    CUDA_OK(cudaFree(QKt_d));
    CUDA_OK(cudaFree(output_d));
}

void softmax_attention_cublas(
    const float* Q, 
    const float* K, 
    const float* V, 
    float* output, 
    int M, int N, int d) 
{
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Allocate device memory
    float *Q_d, *K_d, *Kt_d, *V_d, *QKt_d, *output_d;
    CUDA_OK(cudaMalloc(&Q_d, M * d * sizeof(float)));
    CUDA_OK(cudaMalloc(&K_d, N * d * sizeof(float)));
    CUDA_OK(cudaMalloc(&Kt_d, d * N * sizeof(float)));
    CUDA_OK(cudaMalloc(&V_d, N * d * sizeof(float)));
    CUDA_OK(cudaMalloc(&QKt_d, M * N * sizeof(float)));
    CUDA_OK(cudaMalloc(&output_d, M * d * sizeof(float)));
    
    // Copy data to device
    CUDA_OK(cudaMemcpy(Q_d, Q, M * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(K_d, K, N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(V_d, V, N * d * sizeof(float), cudaMemcpyHostToDevice));
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Step 1: Transpose K using cuBLAS geam
    // Kt[d x N] = K^T[d x N] where K is [N x d]
    // We're working with row-major, but cuBLAS expects column-major
    // So we treat our row-major [N x d] as column-major [d x N], then transpose
    cublasSgeam(handle, 
                CUBLAS_OP_T, CUBLAS_OP_N,
                N, d,                    // output dimensions: N x d (becomes d x N in row-major)
                &alpha, K_d, d,          // K treated as [d x N] in col-major
                &beta, K_d, N,           // dummy
                Kt_d, N);                // output stride
    
    // Step 2: Matrix multiply Q @ K^T using cuBLAS
    // QKt[M x N] = Q[M x d] @ Kt[d x N]
    // In cuBLAS col-major thinking: C = A*B where
    // A is [d x M] in col-major (our Q transposed)
    // B is [N x d] in col-major (our Kt transposed)
    // C is [N x M] in col-major (our result transposed)
    // We use: C^T = B^T * A^T, so QKt = Kt^T * Q^T
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, d,
                &alpha,
                Kt_d, N,      // Kt in row-major [d x N] is like [N x d] in col-major
                Q_d, d,       // Q in row-major [M x d] is like [d x M] in col-major  
                &beta,
                QKt_d, N);    // QKt in row-major [M x N] is like [N x M] in col-major
    
    // // Step 3: Divide by sqrt(d)
    // dim3 block = dim3(256);
    // dim3 grid = dim3(CEIL_DIV(M * N, block.x));
    // matrix_divide<<<grid, block>>>(QKt_d, sqrtf(d), M, N);
    // CUDA_OK(cudaPeekAtLastError());
    // CUDA_OK(cudaDeviceSynchronize());
    
    // // Step 4: Row-wise softmax
    // for (int i = 0; i < M; ++i) {
    //     softmax(QKt_d + i * N, QKt_d + i * N, N);
    // }

    // batched divide softmax
    dim3 block = dim3(256);
    dim3 grid = dim3(M);
    batched_divide_softmax_kernel<<<grid, block>>>(QKt_d, sqrtf(d), M, N);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    
    // Step 5: Matrix multiply softmax(QKt) @ V
    // output[M x d] = QKt[M x N] @ V[N x d]
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                d, M, N,
                &alpha,
                V_d, d,       // V in row-major [N x d] is like [d x N] in col-major
                QKt_d, N,     // QKt in row-major [M x N] is like [N x M] in col-major
                &beta,
                output_d, d); // output in row-major [M x d] is like [d x M] in col-major
    
    // Copy result back to host
    CUDA_OK(cudaMemcpy(output, output_d, M * d * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_OK(cudaFree(Q_d));
    CUDA_OK(cudaFree(K_d));
    CUDA_OK(cudaFree(Kt_d));
    CUDA_OK(cudaFree(V_d));
    CUDA_OK(cudaFree(QKt_d));
    CUDA_OK(cudaFree(output_d));
    cublasDestroy(handle);
}

int main() {
    printf("Testing correctness with small array...\n");
    int M=2, d=4, N=3;
    // Example taken from README.md
    float Q[M][d] = {{1.0f, 0.0f, 0.0f, 0.0f},
                     {0.0f, 1.0f, 0.0f, 0.0f}};
    float K[N][d] = {{1.0f, 0.0f, 0.0f, 0.0f},
                     {0.0f, 1.0f, 0.0f, 0.0f},
                     {0.0f, 0.0f, 1.0f, 0.0f}};
    float V[N][d] = {{1.0f, 2.0f, 3.0f, 4.0f},
                     {5.0f, 6.0f, 7.0f, 8.0f},
                     {9.0f, 10.0f, 11.0f, 12.0f}};
    float output[M][d];
    float expected[2][4] = {
        {4.29f, 5.29f, 6.29f, 7.29f},
        {5.0f, 6.0f, 7.0f, 8.0f}
    };
    softmax_attention((const float*)Q, (const float*)K, (const float*)V, (float*)output, M, N, d);
    float eps = 1e-2f;
    int failed = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < d; ++j) {
            if (fabsf(output[i][j] - expected[i][j]) > eps) {
                printf("Error at index %d, %d: expected %.8f, got %.8f\n", i, j, expected[i][j], output[i][j]);
                failed = 1;
            }
        }
    }
    if (failed) printf("✓ Result is incorrect\n");
    else printf("✓ Result is correct ");
    return 0;
}
