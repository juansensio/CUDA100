// #include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        C[index] = A[index] + B[index];
    }
}

int main() {
    int N = 4;
    float A[] = {1.0, 2.0, 3.0, 4.0};
    float B[] = {5.0, 6.0, 7.0, 8.0};
    float C[N];
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, N * sizeof(float));
    cudaMalloc((void**)&B_d, N * sizeof(float));
    cudaMalloc((void**)&C_d, N * sizeof(float));
    cudaMemcpy(A_d, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N * sizeof(float), cudaMemcpyHostToDevice);

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, C_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify the result
    printf("A = [%.1f, %.1f, %.1f, %.1f]\n", A[0], A[1], A[2], A[3]);
    printf("B = [%.1f, %.1f, %.1f, %.1f]\n", B[0], B[1], B[2], B[3]);
    printf("C = [%.1f, %.1f, %.1f, %.1f]\n", C[0], C[1], C[2], C[3]);
    
    // Check if result is correct
    float expected[] = {6.0, 8.0, 10.0, 12.0};
    int correct = 1;
    for (int i = 0; i < N; i++) {
        if (C[i] != expected[i]) {
            correct = 0;
            printf("Error at index %d: expected %.1f, got %.1f\n", i, expected[i], C[i]);
        }
    }
    if (correct) {
        printf("✓ Result is correct!\n");
    }
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    return 0;
}
