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

__global__ void convolution_1d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ kernel, 
    float* __restrict__ output,
    int input_size, int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size - kernel_size + 1) {
        float sum = 0.0f;
        for (int i = 0; i < kernel_size; ++i) {
            sum += input[idx + i] * kernel[i];
        }
        output[idx] = sum;
    }
}


int main() {
    printf("Testing 1D convolution...\n");
    float input[]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float kernel[]  = {1.0f, 0.0f, -1.0f};
    float expected[] = {-2.0f, -2.0f, -2.0f};
    int input_size = 5;
    int kernel_size = 3;
    int output_size = input_size - kernel_size + 1;
    float output[output_size];
    
    float* input_d, *kernel_d, *output_d;
    CUDA_OK(cudaMalloc((void**)&input_d, input_size * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&kernel_d, kernel_size * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&output_d, output_size * sizeof(float)));
    CUDA_OK(cudaMemcpy(input_d, input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(kernel_d, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_d, kernel_d, output_d, input_size, kernel_size);

    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(output, output_d, output_size * sizeof(float), cudaMemcpyDeviceToHost));
        
    // Verify the result
    for (int i = 0; i < output_size; ++i) { 
        if (output[i] != expected[i]) {
            printf("Error at index %d: expected %.1f, got %.1f\n", i, expected[i], output[i]);
            CUDA_OK(cudaFree(input_d));
            CUDA_OK(cudaFree(kernel_d));
            CUDA_OK(cudaFree(output_d));
            return EXIT_FAILURE;
        }
    }
    printf("✓ Result is correct!\n");
    CUDA_OK(cudaFree(input_d));
    CUDA_OK(cudaFree(kernel_d));
    CUDA_OK(cudaFree(output_d));
    // benchmark();
    return 0;
}
