__global__ void matrix_divide(float* A, float d, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        A[idx] /= d;
    }
};