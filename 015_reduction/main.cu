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

// no podemos sincronizar a nivel global
// cada block se calcula su valor
// lanzamos kernel de manera recursiva hasta que solo quede un block
__global__ void reduction_kernel(const float* input, float* output, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input element or 0 if out-of-bounds
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// void benchmark() {
//     // Range for N: 1 to 10,000 (see constraint)
//     int sizes[] = {100, 1000, 10000, 100000};
//     const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
//     const int num_runs = 8;

//     printf("\nBenchmarking silu_kernel for various N...\n");
//     printf("%-11s %-14s %-14s %-14s\n", "N", "Time (ms)", "BW (GB/s)", "TFLOPs");

//     for (int idx = 0; idx < num_sizes; ++idx) {
//         int N = sizes[idx];
//         int halfN = N / 2;
//         size_t bytes = N * sizeof(float);

//         // Allocate and initialize input array with floats in [-100.0, 100.0]
//         float *input = (float*)malloc(bytes);
//         for (int i = 0; i < N; ++i) {
//             input[i] = ((float)rand()/(float)RAND_MAX) * 200.0f - 100.0f;
//         }

//         float *input_d = nullptr, *output_d = nullptr;
//         CUDA_OK(cudaMalloc((void**)&input_d, bytes));
//         CUDA_OK(cudaMalloc((void**)&output_d, halfN * sizeof(float)));
//         CUDA_OK(cudaMemcpy(input_d, input, bytes, cudaMemcpyHostToDevice));

//         dim3 block(256);
//         dim3 grid((halfN + block.x - 1) / block.x);

//         // --- kernel timing ---
//         float ms_total = 0.0f;
//         cudaEvent_t start, stop;
//         cudaEventCreate(&start);
//         cudaEventCreate(&stop);

//         // Warm-up
//         swiglu_kernel<<<grid, block>>>(input_d, output_d, halfN);
//         CUDA_OK(cudaDeviceSynchronize());

//         for (int run = 0; run < num_runs; ++run) {
//             cudaEventRecord(start);
//             swiglu_kernel<<<grid, block>>>(input_d, output_d, halfN);
//             cudaEventRecord(stop);
//             cudaEventSynchronize(stop);
//             float ms = 0.0f;
//             cudaEventElapsedTime(&ms, start, stop);
//             ms_total += ms;
//         }
//         ms_total /= num_runs;

//         // Calculate memory bandwidth (read + write = 2 * N * sizeof(float))
//         double total_bytes = 2.0 * halfN * sizeof(float);
//         double bandwidth_gbps = total_bytes / (ms_total * 1e6); // GB/s

//         // Compute FLOPs for SiLU operation
//         // Each output: 1 mul (*x), 2 add/sub, 1 exp, 1 div, and 1 more mul (x*sigmoid) = about 5 FLOP
//         // Realistically, exp and div counted as one each, so a rough estimate is 5 FLOPs per element
//         double flops_per_element = 5.0; 
//         double total_flops = halfN * flops_per_element;
//         double tflops = (total_flops / (ms_total * 1e-3)) / 1e12; // TFLOPs

//         printf("%-11d %-14.3f %-14.2f %-14.4f\n", halfN, ms_total, bandwidth_gbps, tflops);

//         free(input);
//         CUDA_OK(cudaFree(input_d));
//         CUDA_OK(cudaFree(output_d));
//         cudaEventDestroy(start);
//         cudaEventDestroy(stop);
//     }
// }

// Recursive reduction on GPU
float reduction_recursive(const float* input, int N) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    float* output_d;
    CUDA_OK(cudaMalloc((void**)&output_d, blocksPerGrid * sizeof(float)));
    
    // Launch kernel with shared memory
    reduction_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        input, output_d, N
    );
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    
    // Base case: if only one block, we're done
    if (blocksPerGrid == 1) {
        float result;
        CUDA_OK(cudaMemcpy(&result, output_d, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaFree(output_d));
        return result;
    }
    
    // Recursive case: reduce the partial sums
    float result = reduction_recursive(output_d, blocksPerGrid);
    CUDA_OK(cudaFree(output_d));
    return result;
}

int main() {
    printf("Testing correctness with small array...\n");
    int N = 8;
    float input[N] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float expected = 36.0;

    float *input_d;
    CUDA_OK(cudaMalloc((void**)&input_d, N * sizeof(float)));
    CUDA_OK(cudaMemcpy(input_d, input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    float result = reduction_recursive(input_d, N);

    // Verify the result
    float eps = 1e-5;
    if (fabsf(result - expected) > eps) {
        printf("Error: expected %.6f, got %.6f\n", expected, result);
        CUDA_OK(cudaFree(input_d));
        return EXIT_FAILURE;
    }
    printf("✓ Result is correct: %.6f\n", result);
    // benchmark();
    return 0;
}
