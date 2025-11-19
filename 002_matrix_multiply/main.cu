#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <string.h>

#define CUDA_OK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

inline unsigned int CEIL_DIV(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

// Forward declarations
typedef void (*KernelFunc)(const float* __restrict__, const float* __restrict__, float* __restrict__, int, int, int);

// Kernel descriptor structure
typedef struct {
    const char* name;
    KernelFunc kernel;
    int is_cublas;  // 1 if cuBLAS, 0 if custom kernel
} KernelDescriptor;

__global__ void matrix_multiply_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C, 
    int M, int N, int K
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float value = 0.0f;
        for (int i = 0; i < K; ++i) {
            value += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

__global__ void matrix_multiply_coalescing(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C, 
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float value = 0.0f;
        for (int i = 0; i < K; ++i) {
            value += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

#define BLOCK_SIZE 16

__global__ void matrix_multiply_shared_memory(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    float sum = 0.0f;
    // Number of tiles along K dimension
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        int tiled_k_A = t * BLOCK_SIZE + threadIdx.x; // column in A
        int tiled_k_B = t * BLOCK_SIZE + threadIdx.y; // row in B
        // Load tile from A into shared memory
        if (row < M && tiled_k_A < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tiled_k_A];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Load tile from B into shared memory
        if (tiled_k_B < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[tiled_k_B * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        // Compute partial sum for current tile
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    // Only write valid elements
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

#define BM 64      // block tile in M dimension   (rows of C per block)
#define BN 64      // block tile in N dimension   (cols of C per block)
#define BK 8      // depth of each K tile
#define TM 8       // rows of C per thread
#define TN 8      // cols of C per thread

typedef unsigned int uint;

__global__ void matrix_multiply_tiling(
    const float* __restrict__ A,  // [M x K]
    const float* __restrict__ B,  // [K x N]
    float* __restrict__ C,        // [M x N]
    int M, int N, int K
) {
    const uint cRow = blockIdx.y;   // block tile index along M
    const uint cCol = blockIdx.x;   // block tile index along N

    // Total results per block tile = BM * BN
    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile   = totalResultsBlocktile / (TM * TN);  // 256

    // 1D thread index in block
    const uint tid = threadIdx.x;

    // Thread's 2D position in the grid of TM×TN microtiles
    const uint threadCol = tid % (BN / TN);  // 0 .. (BN/TN - 1)
    const uint threadRow = tid / (BN / TN);  // 0 .. (BM/TM - 1)

    // Shared memory tiles
    __shared__ float As[BM * BK];  // BM x BK
    __shared__ float Bs[BK * BN];  // BK x BN

    // Move A,B,C pointers to start of this block tile
    A += cRow * BM * K;            // advance rows in A
    B += cCol * BN;                // advance columns in B
    C += cRow * BM * N + cCol * BN;

    // Indices used for cooperative loads from GMEM to SMEM
    const uint innerRowA = tid / BK;
    const uint innerColA = tid % BK;      // *** IMPORTANT: % BK, not / BK ***
    const uint strideA   = numThreadsBlocktile / BK;

    const uint innerRowB = tid / BN;
    const uint innerColB = tid % BN;
    const uint strideB   = numThreadsBlocktile / BN;

    // Per-thread accumulator and register caches
    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    // Outer-most loop over K dimension in steps of BK
    for (uint bkIdx = 0; bkIdx < (uint)K; bkIdx += BK) {
        // ---- Load A tile (BM x BK) into shared memory ----
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            uint aRow = innerRowA + loadOffset;
            uint aCol = innerColA;
            uint gRow = aRow;
            uint gK   = bkIdx + aCol;

            if (gRow < (uint)M && gK < (uint)K) {
                As[aRow * BK + aCol] = A[gRow * K + gK];
            } else {
                As[aRow * BK + aCol] = 0.0f;
            }
        }

        // ---- Load B tile (BK x BN) into shared memory ----
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            uint bRow = innerRowB + loadOffset;
            uint bCol = innerColB;
            uint gK   = bkIdx + bRow;
            uint gCol = bCol;

            if (gK < (uint)K && (cCol * BN + gCol) < (uint)N) {
                Bs[bRow * BN + bCol] = B[gK * N + gCol];
            } else {
                Bs[bRow * BN + bCol] = 0.0f;
            }
        }

        __syncthreads();

        // ---- Compute partial results using this K tile ----
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Bring one column of As and one row of Bs into registers
            for (uint i = 0; i < TM; ++i) {
                uint rowInTile = threadRow * TM + i;
                regM[i] = As[rowInTile * BK + dotIdx];
            }
            for (uint j = 0; j < TN; ++j) {
                uint colInTile = threadCol * TN + j;
                regN[j] = Bs[dotIdx * BN + colInTile];
            }

            // Outer product regM x regN, accumulate into threadResults
            for (uint i = 0; i < TM; ++i) {
                for (uint j = 0; j < TN; ++j) {
                    threadResults[i * TN + j] += regM[i] * regN[j];
                }
            }
        }

        __syncthreads();
    }

    // ---- Write back results (with bounds checks) ----
    for (uint i = 0; i < TM; ++i) {
        uint rowInTile = threadRow * TM + i;
        uint gRow = cRow * BM + rowInTile;
        if (gRow >= (uint)M) continue;

        for (uint j = 0; j < TN; ++j) {
            uint colInTile = threadCol * TN + j;
            uint gCol = cCol * BN + colInTile;
            if (gCol >= (uint)N) continue;

            C[(rowInTile) * N + colInTile] = threadResults[i * TN + j];
        }
    }
}


// Validation function
int validate_kernel(const KernelDescriptor* kernel, 
                    const float* A, const float* B, 
                    float* C_result, const float* C_expected,
                    int M, int N, int K) {
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    
    float *A_d, *B_d, *C_d;
    CUDA_OK(cudaMalloc((void**)&A_d, bytes_A));
    CUDA_OK(cudaMalloc((void**)&B_d, bytes_B));
    CUDA_OK(cudaMalloc((void**)&C_d, bytes_C));
    
    CUDA_OK(cudaMemcpy(A_d, A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(B_d, B, bytes_B, cudaMemcpyHostToDevice));
    
    if (kernel->is_cublas) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, B_d, N, A_d, K, &beta, C_d, N);
        cublasDestroy(handle);
    } else {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x,
                (M + block.y - 1) / block.y);
        kernel->kernel<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
        CUDA_OK(cudaPeekAtLastError());
    }
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(C_result, C_d, bytes_C, cudaMemcpyDeviceToHost));
    // Verify results
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (fabsf(C_result[i*N + j] - C_expected[i*N + j]) > 1e-6f) {
                CUDA_OK(cudaFree(A_d));
                CUDA_OK(cudaFree(B_d));
                CUDA_OK(cudaFree(C_d));
                return 0;  // Validation failed
            }
        }
    }
    CUDA_OK(cudaFree(A_d));
    CUDA_OK(cudaFree(B_d));
    CUDA_OK(cudaFree(C_d));
    return 1;  // Validation passed
}

// Benchmark result structure
typedef struct {
    float ms;
    float gflops;
    float speedup;  // relative to baseline
} BenchmarkResult;

// Benchmark a single kernel
BenchmarkResult benchmark_kernel(const KernelDescriptor* kernel,
                                 const float* A, const float* B,
                                 int M, int N, int K,
                                 int num_runs, float baseline_ms) {
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    
    float *A_d, *B_d, *C_d;
    CUDA_OK(cudaMalloc((void**)&A_d, bytes_A));
    CUDA_OK(cudaMalloc((void**)&B_d, bytes_B));
    CUDA_OK(cudaMalloc((void**)&C_d, bytes_C));
    
    CUDA_OK(cudaMemcpy(A_d, A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(B_d, B, bytes_B, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    BenchmarkResult result = {0.0f, 0.0f, 0.0f};
    
    // Warm-up
    if (kernel->is_cublas) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, B_d, N, A_d, K, &beta, C_d, N);
        CUDA_OK(cudaDeviceSynchronize());
        cublasDestroy(handle);
    } else {
        if (strcmp(kernel->name, "Tiling") == 0) {
            dim3 block((BM * BN) / (TM * TN));
            dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
            kernel->kernel<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
        } else {
            dim3 block(16, 16);
            dim3 grid((N + block.x - 1) / block.x,
                    (M + block.y - 1) / block.y);
            kernel->kernel<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
        }
        CUDA_OK(cudaDeviceSynchronize());
    }
    // Benchmark runs
    float total_ms = 0.0f;
    for (int run = 0; run < num_runs; ++run) {
        cudaEventRecord(start);
        if (kernel->is_cublas) {
            cublasHandle_t handle;
            cublasCreate(&handle);
            float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, B_d, N, A_d, K, &beta, C_d, N);
            cublasDestroy(handle);
        } else {
            if (strcmp(kernel->name, "Tiling") == 0) {
                dim3 block((BM * BN) / (TM * TN));
                dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
                kernel->kernel<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
            } else {
                dim3 block(16, 16);
                dim3 grid((N + block.x - 1) / block.x,
                             (M + block.y - 1) / block.y);
                kernel->kernel<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
            }
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float tmp;
        cudaEventElapsedTime(&tmp, start, stop);
        total_ms += tmp;
    }
    result.ms = total_ms / num_runs;
    result.gflops = (2.0 * M * N * K) / (result.ms * 1e6);
    if (baseline_ms > 0.0f) {
        result.speedup = baseline_ms / result.ms;  // Fixed: baseline/measured
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_OK(cudaFree(A_d));
    CUDA_OK(cudaFree(B_d));
    CUDA_OK(cudaFree(C_d));
    return result;
}

// Kernel registry - easy to add new kernels here
KernelDescriptor kernels[] = {
    {"cuBLAS", NULL, 1},
    {"Naive", matrix_multiply_naive, 0},
    {"Coalescing", matrix_multiply_coalescing, 0},
    {"Shared Memory", matrix_multiply_shared_memory, 0},
    {"Tiling", matrix_multiply_tiling, 0}
};

const int num_kernels = sizeof(kernels) / sizeof(kernels[0]);

void benchmark() {
    printf("\nBenchmarking kernels for various sizes...\n");
    printf("%-15s%-12s%-12s%-12s%-12s\n", 
        "Kernel", "Size", "Time (ms)", "TFLOPs", "Speedup");
    printf("%s\n", "------------------------------------------------------------");

    int sizes[] = {512, 1024, 2048, 4096};
    const int num_sizes = sizeof(sizes)/sizeof(sizes[0]);
    const int num_runs = 5;

    for (int idx = 0; idx < num_sizes; ++idx) {
        int M = sizes[idx], N = sizes[idx], K = sizes[idx];
        size_t bytes_A = M * K * sizeof(float);
        size_t bytes_B = K * N * sizeof(float);

        float *A = (float*)malloc(bytes_A);
        float *B = (float*)malloc(bytes_B);

        for (int i = 0; i < M * K; ++i) A[i] = 1.0f;
        for (int i = 0; i < K * N; ++i) B[i] = 1.0f;

        float baseline_ms = 0.0f;

        // Benchmark each kernel and print results immediately
        for (int k = 0; k < num_kernels; ++k) {
            BenchmarkResult result = benchmark_kernel(&kernels[k], A, B, M, N, K, num_runs, 0.0f);
            
            // First kernel (cuBLAS) becomes the baseline
            if (k == 0) {
                baseline_ms = result.ms;
            }
            
            // Calculate speedup: baseline_ms / result.ms
            float speedup = (k > 0 && baseline_ms > 0.0f) ? baseline_ms / result.ms : 1.0f;
            
            // Print result immediately
            printf("%-15s%-12d%-12.4f%-12.4f", kernels[k].name, sizes[idx], result.ms, result.gflops / 1000.0f);
            if (k > 0) {
                printf("%-12.2f", speedup);
            } else {
                printf("%-12s", "-");
            }
            printf("\n");
            fflush(stdout);  // Ensure output appears immediately
        }
        
        printf("\n");  // Blank line between sizes
        free(A);
        free(B);
    }
}

int main() {
    // Setup test matrices
    printf("Testing correctness with small matrices...\n");
    int N = 2;
    int M = 2;
    int K = 3;
    float A[M][K] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}; 
    float B[K][N] = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}}; 
    float expected[M][N] = {{58.0, 64.0}, {139.0, 154.0}};
    
    float* C_result = (float*)malloc(M * N * sizeof(float));
    if (C_result == NULL) {
        fprintf(stderr, "Failed to allocate host memory for C_result\n");
        return EXIT_FAILURE;
    }
    
    // Validate each kernel (skip cuBLAS for validation, use it as reference)
    // First compute reference using cuBLAS
    KernelDescriptor cublas_ref = {"cuBLAS", NULL, 1};
    if (!validate_kernel(&cublas_ref, (float*)A, (float*)B, C_result, (float*)expected, M, N, K)) {
        printf("✗ cuBLAS reference validation failed!\n");
        free(C_result);
        return EXIT_FAILURE;
    }
    
    // Validate custom kernels (skip cuBLAS in the loop)
    for (int k = 1; k < num_kernels; ++k) {
        if (!validate_kernel(&kernels[k], (float*)A, (float*)B, C_result, (float*)expected, M, N, K)) {
            printf("✗ %s kernel validation failed!\n", kernels[k].name);
            free(C_result);
            return EXIT_FAILURE;
        }
        printf("✓ %s kernel result is correct!\n", kernels[k].name);
        fflush(stdout);
    }
    
    free(C_result);
    
    // Run benchmarks
    benchmark();
    return 0;
}
