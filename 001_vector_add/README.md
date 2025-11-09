# Vector Addition

## Problem Description

Write a program that performs element-wise addition of two vectors (`A` and `B`) containing 32-bit floating point numbers, utilizing the GPU. The two input vectors must have the same length, and the output should be a single vector (`C`) where each element is the sum of the corresponding elements in `A` and `B`.

**Requirements:**
- Do not use any external libraries.
- The `solve` function signature must not be changed.
- Store the final result in vector `C`.

**Examples:**

- *Example 1*  
  Input:  
  &nbsp;&nbsp;A = [1.0, 2.0, 3.0, 4.0]  
  &nbsp;&nbsp;B = [5.0, 6.0, 7.0, 8.0]  
  Output:  
  &nbsp;&nbsp;C = [6.0, 8.0, 10.0, 12.0]

- *Example 2*  
  Input:  
  &nbsp;&nbsp;A = [1.5, 1.5, 1.5]  
  &nbsp;&nbsp;B = [2.3, 2.3, 2.3]  
  Output:  
  &nbsp;&nbsp;C = [3.8, 3.8, 3.8]

**Constraints:**
- The input vectors `A` and `B` must have the same length.
- $1 \leq N \leq 100,\!000,\!000$, where $N$ is the length of `A` and `B`.

## Solution

Here’s a clear walkthrough of what this CUDA program does, why it works, the key steps, and how you could improve it.

# What it does

* Adds two float vectors element-wise on the GPU: **C[i] = A[i] + B[i]**.
* Host (CPU) code allocates/copies data to device (GPU), launches a kernel, copies the result back, and verifies it.

# Why it works (CUDA model in a nutshell)

* The kernel `vector_add` runs in **many threads**. Each thread computes one index:

  ```c
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) C[index] = A[index] + B[index];
  ```
* Threads are grouped into blocks (`blockDim.x` threads per block), and blocks form a grid (`blocksPerGrid`). The mapping above gives each thread a unique global index.
* Because `A`, `B`, and `C` are contiguous arrays, **adjacent threads access adjacent elements**, which is good for **coalesced global memory access** (fast on GPUs).

# Important steps in the code

1. **Problem size & host data**

   ```c
   int N = 4;
   float A[] = {1,2,3,4}, B[] = {5,6,7,8}, C[N];
   ```

2. **Launch configuration**

   ```c
   int threadsPerBlock = 256;
   int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
   ```

   This ensures enough threads to cover `N` elements (ceiling division).

3. **Device memory allocation**

   ```c
   float *A_d, *B_d, *C_d;
   cudaMalloc(&A_d, N*sizeof(float)); // similar for B_d, C_d
   ```

4. **Host→Device copies**

   ```c
   cudaMemcpy(A_d, A, N*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(B_d, B, N*sizeof(float), cudaMemcpyHostToDevice);
   ```

5. **Kernel launch & (explicit) sync**

   ```c
   vector_add<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);
   cudaDeviceSynchronize();
   ```

   (Note: a following `cudaMemcpy(...DeviceToHost)` also synchronizes, but `cudaDeviceSynchronize()` helps catch kernel errors earlier if you check them.)

6. **Device→Host copy & verification**

   ```c
   cudaMemcpy(C, C_d, N*sizeof(float), cudaMemcpyDeviceToHost);
   // compare with expected
   ```

7. **Cleanup**

   ```c
   cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
   ```

# Potential improvements (from “works” to “robust & scalable”)

### 1) Check all CUDA errors

Right now, failures would go unnoticed. Wrap calls with a macro:

```c
#define CUDA_OK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} while(0)

// Use like:
CUDA_OK(cudaMalloc(&A_d, N*sizeof(float)));
vector_add<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);
CUDA_OK(cudaPeekAtLastError()); // launch errors
CUDA_OK(cudaDeviceSynchronize()); // runtime errors
```

### 2) Use a grid-stride loop (scales to any N and launch size)

```c
__global__ void vector_add(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C, int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < N;
       i += blockDim.x * gridDim.x) {
    C[i] = A[i] + B[i];
  }
}
```

* Works well even if `N` >> number of total threads.
* `__restrict__` may help the compiler optimize memory loads.

### 3) Pick a sensible launch for tiny/huge N

* For **N=4**, `threadsPerBlock=256` is overkill. You could set:

  ```c
  int tpb = 128; // or min(256, nextPow2(N))
  int bpg = (N + tpb - 1) / tpb;
  ```
* For **large N**, consider computing an occupancy-friendly `tpb` (e.g., 128–1024 depending on the GPU) and capping `bpg` (e.g., `min(bpg, 32*sm_count)`).

### 4) Avoid VLA in C++ host code

`float C[N];` is a variable-length array (VLA) and not standard C++. Prefer:

```c++
std::vector<float> C(N);
```

(or stick to C99, or allocate fixed-size when known).

### 5) Prefer tolerant float checks

Equality is fine here (exact decimals), but generally:

```c
float eps = 1e-6f;
if (fabsf(C[i] - expected[i]) > eps) { /* report */ }
```

### 6) Simplify with Unified Memory (for learning/prototyping)

Removes explicit copies:

```c
float *A, *B, *C;
cudaMallocManaged(&A, N*sizeof(float));
cudaMallocManaged(&B, N*sizeof(float));
cudaMallocManaged(&C, N*sizeof(float));
// initialize A,B on host, launch kernel, then cudaDeviceSynchronize(); use C.
```

(Explicit `cudaMemPrefetchAsync` can improve performance on newer GPUs.)

### 7) Use pinned (page-locked) host memory for faster transfers

```c
float *A_h, *B_h, *C_h;
cudaMallocHost(&A_h, N*sizeof(float)); // pinned
// initialize, memcpy H<->D can be faster; enables async copies with streams.
```

### 8) Overlap transfers and compute with streams (when N is large)

Chunk the work and pipeline `cudaMemcpyAsync` with kernel launches in multiple streams.

### 9) Leverage libraries

For a simple transform:

```c++
thrust::device_vector<float> A(A_h, A_h+N), B(B_h, B_h+N), C(N);
thrust::transform(A.begin(), A.end(), B.begin(), C.begin(), thrust::plus<float>());
```

Cleaner and often well-optimized.

### 10) Micro-optimizations (when it matters)

* Load/store as `float4` if `N` is large and data is aligned, to reduce instruction count.
* Consider `-use_fast_math` (with care).
* Profile with `nsys`/`nvprof`/`nsight` and check occupancy/memory throughput.

### 11) Housekeeping

* Uncomment the proper include:

  ```c
  #include <cuda_runtime.h>
  #include <cstdio>
  ```
* Compile with `nvcc`, e.g.:

  ```
  nvcc -O2 -arch=sm_80 vec_add.cu -o vec_add
  ```

  (Change `sm_80` to match your GPU.)
* Use `cuda-memcheck ./vec_add` during development.
