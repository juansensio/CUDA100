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