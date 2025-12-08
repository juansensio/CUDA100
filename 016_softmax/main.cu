#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define CUDA_OK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void exp_subtract_max_kernel(const float* input, float* output, float max_val, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = expf(input[idx] - max_val);
    }
}

__global__ void normalize_kernel(float* output, float sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] /= sum;
    }
}

void softmax(const float* input_d, float* output_d, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Step 1: find max
    thrust::device_ptr<const float> dev_ptr_input(input_d);
    float h_max = thrust::reduce(dev_ptr_input, dev_ptr_input + N, -INFINITY, thrust::maximum<float>());

    // step 2: compute exp(input - max) for numerical stability
    // probar ambas a ver que va mejor
    // thrust::device_ptr<float> dev_ptr_output(output_d);
    // thrust::transform(dev_ptr_output, dev_ptr_output + N, dev_ptr_output, [h_max](float x) { return exp(x - h_max); });
    exp_subtract_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_d, output_d, h_max, N);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());

    // Step 3: Sum all exponentiated values using Thrust
    thrust::device_ptr<float> dev_ptr_output(output_d);
    float h_sum = thrust::reduce(dev_ptr_output, dev_ptr_output + N, 0.0f, thrust::plus<float>());

    // Step 4: Normalize by dividing each element by sum
    normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(output_d, h_sum, N);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
}

int main() {
    printf("Testing correctness with small array...\n");
    int N = 3;
    float input[N] = {1.0, 2.0, 3.0};
    float expected[N] = {0.090, 0.244, 0.665}; // approx

    float *input_d, *output_d;
    CUDA_OK(cudaMalloc((void**)&input_d, N * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&output_d, N * sizeof(float)));
    CUDA_OK(cudaMemcpy(input_d, input, N * sizeof(float), cudaMemcpyHostToDevice));

    float* result = (float*)malloc(N * sizeof(float));
    softmax(input_d, output_d, N);
    CUDA_OK(cudaMemcpy(result, output_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(input_d));
    CUDA_OK(cudaFree(output_d));

    float eps = 1e-3f;
    int failed = 0;

    for (int i = 0; i < N; ++i) {
        if (fabsf(result[i] - expected[i]) > eps) {
            printf("Error at index %d: expected %.8f, got %.8f\n", i, expected[i], result[i]);
            failed = 1;
        }
    }

    if (failed) return EXIT_FAILURE;
    printf("✓ Result is correct: ");
    for (int i = 0; i < N; ++i) printf("%.6f ", result[i]);
    printf("\n");
    free(result);
    // benchmark();
    return 0;
}
