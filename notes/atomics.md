# Atomics

Atomics are used to perform atomic operations on shared memory. They are used to ensure that the operations are performed atomically, i.e. without any other threads interfering. Values are not accessed concurrently, so no data races occur. A thread cannot read a value while another thread is writing to it.

Slow down the execution of the kernel but memory safe.

## Atomic Functions in CUDA

CUDA provides a set of built-in atomic functions that enable safe concurrent updates to a shared variable from multiple threads without data races. Atomic operations guarantee that read-modify-write sequences are completed by one thread at a time.

Here are commonly used atomic functions on integer types:

- `atomicAdd(address, val)`: Atomically adds `val` to the variable at `address`. Returns the old value.
    - **Example:**  
      ```c
      atomicAdd(&counter, 1); // increments counter by 1 atomically
      ```
- `atomicSub(address, val)`: Atomically subtracts `val` from the variable at `address`. Returns the old value.
- `atomicMin(address, val)`: Atomically sets the variable at `address` to the minimum of its current value and `val`. Returns the old value.
- `atomicMax(address, val)`: Atomically sets the variable at `address` to the maximum of its current value and `val`. Returns the old value.
- `atomicAnd(address, val)`: Atomically performs a bitwise AND with `val` on the variable at `address`. Returns the old value.
- `atomicOr(address, val)`: Atomically performs a bitwise OR with `val`. Returns the old value.
- `atomicXor(address, val)`: Atomically performs a bitwise XOR with `val`. Returns the old value.
- `atomicExch(address, val)`: Atomically replaces the value at `address` with `val`. Returns the old value.
- `atomicCAS(address, compare, val)`: Atomic Compare-And-Swap. If the current value at `address` equals `compare`, set to `val`. Returns the old value regardless.

### How Atomics Work

Atomic operations are slow compared to regular operations because they serialize all writes (and sometimes reads) to the same memory location—they must ensure no race conditions occur. They are nevertheless essential when multiple threads must update or check the same variable.

### Example: Concurrent Count with `atomicAdd`

Suppose you want to count the number of times a value appears in an array (from different threads):

```c
__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (input[idx] == K) {
            atomicAdd(output, 1); // Safely increment count
        }
    }
}
```

If you replaced `atomicAdd(output, 1)` with `(*output)++`, the result would be undefined due to data races.

### Notes

- Atomic operations are available on integers and, in modern CUDA, floating point (with some limitations).
- Atomic functions work on global, shared, and (for some functions) local memory.
- Excessive use of atomics can degrade performance (contention, serialization), so structure your code to minimize atomic usage if possible.

For more details, see [NVIDIA's atomic functions documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions).
