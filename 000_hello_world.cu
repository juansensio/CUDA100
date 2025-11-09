#include <stdio.h>

__global__ void hello_cuda() {
    printf("Hello, World from GPU! (thread %d, block %d)\n", threadIdx.x, blockIdx.x);
}

int main() {
    // Launch the kernel with 1 block and 5 threads
    hello_cuda<<<1, 5>>>();
    cudaDeviceSynchronize(); // Wait for GPU to finish
    return 0;
}
    