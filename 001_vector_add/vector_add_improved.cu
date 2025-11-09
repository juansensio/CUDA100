// Improvement 1: Include proper CUDA headers
// Uncommented cuda_runtime.h to ensure all CUDA runtime functions are properly declared
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Improvement 2: CUDA error checking macro
// This ensures all CUDA API calls are checked for errors. Without this, failures would
// go unnoticed, making debugging very difficult. The macro checks the return value,
// prints a descriptive error message with file and line number, and exits on failure.
#define CUDA_OK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Improvement 3: Grid-stride loop with __restrict__ qualifiers
// Grid-stride loop: Instead of each thread processing exactly one element, threads
// process multiple elements with stride = blockDim.x * gridDim.x. This allows the
// kernel to scale to any problem size N, even if N >> total number of threads.
// Benefits:
// - Works efficiently for both small and very large N
// - Better GPU utilization when N is large
// - More flexible launch configuration
// __restrict__: Tells the compiler that pointers A, B, C don't alias (don't point
// to overlapping memory regions), enabling better compiler optimizations.
__global__ void vector_add(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C, int N) {
    // Grid-stride loop: start at this thread's index, then stride by total threads
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 4;
    float A[] = {1.0, 2.0, 3.0, 4.0};
    float B[] = {5.0, 6.0, 7.0, 8.0};
    
    // Improvement 4: Avoid Variable Length Arrays (VLA)
    // VLAs are not standard C++ and can cause stack overflow for large arrays.
    // Using dynamic allocation with malloc/free is more portable and safer.
    float* C = (float*)malloc(N * sizeof(float));
    if (C == NULL) {
        fprintf(stderr, "Failed to allocate host memory for C\n");
        return EXIT_FAILURE;
    }
    
    // Improvement 5: Adaptive launch configuration
    // For small N (like 4), 256 threads per block is overkill and wastes resources.
    // For large N, we want good occupancy. This chooses a reasonable threadsPerBlock
    // that balances between too few threads (poor occupancy) and too many (register
    // pressure). 128-256 is typically a good range for most GPUs.
    int threadsPerBlock = (N < 128) ? 32 : 128;  // Use fewer threads for small N
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Ensure at least one block (important for very small N)
    if (blocksPerGrid == 0) blocksPerGrid = 1;

    float* A_d, *B_d, *C_d;
    
    // Improvement 2 (continued): Check all CUDA API calls for errors
    CUDA_OK(cudaMalloc((void**)&A_d, N * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&B_d, N * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&C_d, N * sizeof(float)));
    
    CUDA_OK(cudaMemcpy(A_d, A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(B_d, B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel with error checking
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);
    
    // Check for kernel launch errors (synchronous errors are caught here)
    CUDA_OK(cudaPeekAtLastError());
    // Check for asynchronous runtime errors
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaMemcpy(C, C_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify the result
    printf("A = [%.1f, %.1f, %.1f, %.1f]\n", A[0], A[1], A[2], A[3]);
    printf("B = [%.1f, %.1f, %.1f, %.1f]\n", B[0], B[1], B[2], B[3]);
    printf("C = [%.1f, %.1f, %.1f, %.1f]\n", C[0], C[1], C[2], C[3]);
    
    // Improvement 6: Tolerant float comparison
    // Direct equality (==) for floats can fail due to floating-point precision issues,
    // even when values should be equal. Using an epsilon tolerance is more robust.
    // For this example with exact decimals, equality works, but the epsilon approach
    // is more general and handles cases where floating-point arithmetic might introduce
    // small errors.
    float expected[] = {6.0, 8.0, 10.0, 12.0};
    const float eps = 1e-6f;  // Tolerance for float comparison
    int correct = 1;
    for (int i = 0; i < N; i++) {
        if (fabsf(C[i] - expected[i]) > eps) {
            correct = 0;
            printf("Error at index %d: expected %.1f, got %.1f\n", i, expected[i], C[i]);
        }
    }
    if (correct) {
        printf("✓ Result is correct!\n");
    }
    
    // Cleanup with error checking
    CUDA_OK(cudaFree(A_d));
    CUDA_OK(cudaFree(B_d));
    CUDA_OK(cudaFree(C_d));
    
    // Free host memory
    free(C);
    
    return 0;
}
