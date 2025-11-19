#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>

#define CUDA_OK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define TILE_DIM   32
#define BLOCK_ROWS 16

__global__ void matrix_transpose(const float* __restrict__ A,
    float* __restrict__ At,
    int N, int M) {
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // if (row < N && col < M) {
    //     At[col*N + row] = A[row*M + col];
    // }
    // ~500 GB/s

    // Shared memory tile - add 1 to avoid bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; 
    int x = blockIdx.x * TILE_DIM + threadIdx.x;  // column in A
    int y = blockIdx.y * BLOCK_ROWS + threadIdx.y;  // row in A
    // 1) Coalesced read from A into shared memory
    //    Each thread reads multiple elements in steps of BLOCK_ROWS
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int yj = y + j;
        if (x < M && yj < N) {
            tile[threadIdx.y + j][threadIdx.x] = A[yj * M + x];
        }
    }
    // 1 thread takes charge of BLOCK_ROWS elements in A
    __syncthreads();
    // 2) Transpose block index for output
    int xo = blockIdx.y * BLOCK_ROWS + threadIdx.x;  // column in At
    int yo = blockIdx.x * TILE_DIM + threadIdx.y;  // row in At
    // 3) Coalesced write from shared memory to At (transposed)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int yoj = yo + j;
        if (xo < N && yoj < M) {
            // note: tile is indexed transposed: [col][row]
            At[yoj * N + xo] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
};

void benchmark_against_cublas() {
    printf("\nBenchmarking transpose kernel against cuBLAS for various sizes...\n");
    printf("%-10s %-14s %-10s %-7s %-14s %-10s %-7s\n", "N", "Custom (ms)", "GB/s", "%Max", "cuBLAS(ms)", "GB/s", "%Max");
    int sizes[] = {1024, 2048, 4096, 8192};
    const int num_sizes = sizeof(sizes)/sizeof(sizes[0]);
    const int num_runs = 5;
    for (int idx = 0; idx < num_sizes; ++idx) {
        int M = sizes[idx], N = sizes[idx]; // Square for fairness
        size_t bytes_A = M * N * sizeof(float);

        float *A = (float*)malloc(bytes_A);
        float *At_naive = (float*)malloc(bytes_A);
        float *At_cublas = (float*)malloc(bytes_A);

        for (int i = 0; i < M * N; ++i) A[i] = 1.0f;

        float *A_d, *At_d;
        CUDA_OK(cudaMalloc((void**)&A_d, bytes_A));
        CUDA_OK(cudaMalloc((void**)&At_d, bytes_A));

        CUDA_OK(cudaMemcpy(A_d, A, bytes_A, cudaMemcpyHostToDevice));

        dim3 threadsPerBlock(16, 16);
        dim3 gridDim((M + threadsPerBlock.x - 1)/threadsPerBlock.x,
                     (N + threadsPerBlock.y - 1)/threadsPerBlock.y);

        // --- kernel timing ---
        float ms_naive = 0.0f;
        cudaEvent_t start_naive, stop_naive;
        cudaEventCreate(&start_naive);
        cudaEventCreate(&stop_naive);

        // Warm-up
        matrix_transpose<<<gridDim, threadsPerBlock>>>(A_d, At_d, M, N);
        CUDA_OK(cudaDeviceSynchronize());

        for (int run = 0; run < num_runs; ++run) {
            cudaEventRecord(start_naive);
            matrix_transpose<<<gridDim, threadsPerBlock>>>(A_d, At_d, M, N);
            cudaEventRecord(stop_naive);
            cudaEventSynchronize(stop_naive);
            float tmp;
            cudaEventElapsedTime(&tmp, start_naive, stop_naive);
            ms_naive += tmp;
        }
        ms_naive /= num_runs;
        CUDA_OK(cudaMemcpy(At_naive, At_d, bytes_A, cudaMemcpyDeviceToHost));

        CUDA_OK(cudaFree(A_d));
        CUDA_OK(cudaFree(At_d));
        cudaEventDestroy(start_naive);
        cudaEventDestroy(stop_naive);

        // --- cuBLAS timing (using GEAM) ---
        float ms_cublas = 0.0f;
        float *A_d2, *At_d2;
        CUDA_OK(cudaMalloc((void**)&A_d2, bytes_A));
        CUDA_OK(cudaMalloc((void**)&At_d2, bytes_A));
        CUDA_OK(cudaMemcpy(A_d2, A, bytes_A, cudaMemcpyHostToDevice));
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;
        cudaEvent_t start_cublas, stop_cublas;
        cudaEventCreate(&start_cublas);
        cudaEventCreate(&stop_cublas);

        // Warm-up
        cublasSgeam(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N, // transpose A, do not transpose dummy B
            M, N,
            &alpha, A_d2, N,
            &beta,  A_d2, M,
            At_d2, M
        );
        CUDA_OK(cudaDeviceSynchronize());

        for (int run = 0; run < num_runs; ++run) {
            cudaEventRecord(start_cublas);
            cublasSgeam(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                M, N,
                &alpha, A_d2, N,
                &beta,  A_d2, M,
                At_d2, M
            );
            cudaEventRecord(stop_cublas);
            cudaEventSynchronize(stop_cublas);
            float tmp;
            cudaEventElapsedTime(&tmp, start_cublas, stop_cublas);
            ms_cublas += tmp;
        }
        ms_cublas /= num_runs;
        CUDA_OK(cudaMemcpy(At_cublas, At_d2, bytes_A, cudaMemcpyDeviceToHost));

        cublasDestroy(handle);
        CUDA_OK(cudaFree(A_d2));
        CUDA_OK(cudaFree(At_d2));
        cudaEventDestroy(start_cublas);
        cudaEventDestroy(stop_cublas);

        // --- Throughput computation with percentage of theoretical max (936.2 GB/s) ---
        double num_bytes = (double)M * N * sizeof(float) * 2; // read & write
        double gb_naive = num_bytes / (ms_naive * 1e6);
        double gb_cublas = ms_cublas > 0 ? num_bytes / (ms_cublas * 1e6) : 0.0;
        double pct_naive = 100.0 * gb_naive / 936.2; // 936.2 GB/s is the theoretical max bandwidth in a 3090 RTX
        double pct_cublas = gb_cublas > 0 ? 100.0 * gb_cublas / 936.2 : 0.0;
        // double speedup = gb_cublas > 0 ? gb_naive / gb_cublas : 0.0;

        printf("%-10d %-14f %-10.2f %-7.2f %-14f %-10.2f %-7.2f\n",
            M, ms_naive, gb_naive, pct_naive,
            ms_cublas, gb_cublas, pct_cublas);

            
        free(A);
        free(At_naive);
        free(At_cublas);
    }
}

int main() {
    printf("Testing correctness with small matrices...\n");
    int N = 3;
    int M = 2;
    float A[N][M] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}; 
    float expected[M][N] = {{1.0, 3.0, 5.0}, {2.0, 4.0, 6.0}};
    float* At = (float*)malloc(M * N * sizeof(float)); 
    
    float* A_d, *At_d;
    CUDA_OK(cudaMalloc((void**)&A_d, M * N * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&At_d, M * N * sizeof(float)));
    
    CUDA_OK(cudaMemcpy(A_d, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(At_d, At, M * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((M + TILE_DIM - 1) / TILE_DIM,
              (N + TILE_DIM - 1) / TILE_DIM);
    matrix_transpose<<<grid, block>>>(A_d, At_d, N, M);

    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(At, At_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify the result
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            if (At[j*N + i] != expected[j][i]) {
                printf("Error at index %d, %d: expected %.1f, got %.1f\n", i, j, expected[j][i], At[j*N + i]);
                free(At);
                CUDA_OK(cudaFree(A_d));
                CUDA_OK(cudaFree(At_d));
                return EXIT_FAILURE;
            }
        }
    }
    printf("✓ Result is correct!\n");
    free(At);
    CUDA_OK(cudaFree(A_d));
    CUDA_OK(cudaFree(At_d));
    benchmark_against_cublas();
    return 0;
}
