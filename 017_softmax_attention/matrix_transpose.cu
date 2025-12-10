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