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

    // const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx < width * height * 4 && idx % 4 != 3) {
    //     image[idx] = 255 - image[idx];
    // }
    // ~400 GB/s

typedef unsigned int uint;

__global__ void invert_kernel(unsigned char* __restrict__ image, int width, int height) {
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        const uint idx = 4 * (row * width + col);
        image[idx] = 255 - image[idx];
        image[idx + 1] = 255 - image[idx + 1];
        image[idx + 2] = 255 - image[idx + 2];
    }
};

void benchmark() {
    printf("\nBenchmarking color inversion kernel for various sizes...\n");
    printf("%-10s %-14s %-10s\n", "N", "ms", "GB/s");
    int sizes[] = {256, 512, 1024, 2048, 4096};
    const int num_sizes = sizeof(sizes)/sizeof(sizes[0]);
    const int num_runs = 5;
    for (int idx = 0; idx < num_sizes; ++idx) {
        int width = sizes[idx], height = sizes[idx];
        size_t bytes_A = width * height * 4 * sizeof(unsigned char);

        unsigned char *image = (unsigned char*)malloc(bytes_A);
        if (!image) {
            fprintf(stderr, "Host malloc failed\n");
            exit(EXIT_FAILURE);
        }

        // Fill image with dummy data
        for (int i = 0; i < width * height * 4; ++i) image[i] = 1;

        unsigned char *image_d = NULL;
        CUDA_OK(cudaMalloc((void**)&image_d, bytes_A));
        CUDA_OK(cudaMemcpy(image_d, image, bytes_A, cudaMemcpyHostToDevice));

        // dim3 block(256);
        // dim3 grid((width * height * 4 + block.x - 1) / block.x);
        dim3 block(8, 8);
        dim3 grid(
            (width + block.x - 1) / block.x, 
            (height + block.y - 1) / block.y
        );

        // --- kernel timing ---
        float ms = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warm-up
        invert_kernel<<<grid, block>>>(image_d, width, height);
        CUDA_OK(cudaPeekAtLastError());
        CUDA_OK(cudaDeviceSynchronize());

        for (int run = 0; run < num_runs; ++run) {
            cudaEventRecord(start);
            invert_kernel<<<grid, block>>>(image_d, width, height);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float tmp;
            cudaEventElapsedTime(&tmp, start, stop);
            ms += tmp;
        }
        ms /= num_runs;
        CUDA_OK(cudaMemcpy(image, image_d, bytes_A, cudaMemcpyDeviceToHost));

        CUDA_OK(cudaFree(image_d));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // --- Throughput computation with percentage of theoretical max (936.2 GB/s) ---
        double num_bytes = (double)width * height * 4 * sizeof(unsigned char) * 2; // read & write
        double gb = (ms > 0.0) ? num_bytes / (ms * 1e6) : 0.0;
        printf("%-10d %-14f %-10.2f\n", width, ms, gb);

        free(image);
    }
}

int main() {
    printf("Testing correctness with small image...\n");
    int width = 1;
    int height = 2;
    unsigned char image[width * height * 4] = {255, 0, 128, 255, 0, 255, 0, 255};
    unsigned char expected[width * height * 4] = {0, 255, 127, 255, 255, 0, 255, 255};
    unsigned char* image_d;

    CUDA_OK(cudaMalloc((void**)&image_d, width * height * 4 * sizeof(unsigned char)));
    CUDA_OK(cudaMemcpy(image_d, image, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    dim3 block(8, 8);
    dim3 grid(
        (width + block.x - 1) / block.x, 
        (height + block.y - 1) / block.y
    );
    invert_kernel<<<grid, block>>>(image_d, width, height);

    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(image, image_d, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Verify the result
    for (int i = 0; i < width * height * 4; ++i) {
        if (image[i] != expected[i]) {
            printf("Error at index %d: expected %d, got %d\n", i, expected[i], image[i]);
            return EXIT_FAILURE;
        }
    }
    printf("✓ Result is correct!\n");
    CUDA_OK(cudaFree(image_d));
    benchmark();
    return 0;
}
