# Hello World

## 🧠 What this code does

This program prints messages **from the GPU** instead of the CPU.

When you run it, you’ll see output like:

```
Hello, World from GPU! (thread 0, block 0)
Hello, World from GPU! (thread 1, block 0)
Hello, World from GPU! (thread 2, block 0)
Hello, World from GPU! (thread 3, block 0)
Hello, World from GPU! (thread 4, block 0)
```

Each line is printed by a **different GPU thread** that runs the same function (`hello_cuda`), but each with its own **thread index**.

---

## ⚙️ Code explanation line-by-line

```cpp
#include <stdio.h>
```

Includes the standard C I/O library so we can use `printf`.

---

```cpp
__global__ void hello_cuda() {
    printf("Hello, World from GPU! (thread %d, block %d)\n", threadIdx.x, blockIdx.x);
}
```

### `__global__`

* This keyword tells CUDA that the function is a **kernel** — a special function that runs **on the GPU** but is **called from the CPU**.
* The GPU executes many *threads* that run this same function **in parallel**.

### Inside the kernel:

* `threadIdx.x` is a built-in CUDA variable that gives the **thread index within its block**.
* `blockIdx.x` gives the **index of the block** within the grid.

So every thread can identify itself by those values.

---

```cpp
int main() {
    // Launch the kernel with 1 block and 5 threads
    hello_cuda<<<1, 5>>>();
```

### The triple angle brackets `<<< >>>`

This is CUDA syntax to **launch a kernel** on the GPU.

The parameters `<<<1, 5>>>` specify the *execution configuration*:

* **1 block**
* **5 threads per block**

So the GPU will launch **5 parallel threads** that all run `hello_cuda()`.

---

```cpp
    cudaDeviceSynchronize(); // Wait for GPU to finish
    return 0;
}
```

### `cudaDeviceSynchronize()`

* The CPU and GPU run **asynchronously** by default.
* This call tells the CPU to **wait** until the GPU finishes all work before proceeding — in this case, so that all `printf` outputs appear before the program exits.

---

## 🚀 CUDA Concepts Involved

Here are the **core ideas** behind what’s happening:

### 1. Host vs Device

* **Host** = the CPU and its memory (main system RAM)
* **Device** = the GPU and its memory (VRAM)

CUDA programs are “heterogeneous”: part runs on the host (C++ code), part runs on the device (kernels).

---

### 2. Kernels

A **kernel** is a function that runs in parallel across many GPU threads.

* You mark it with `__global__`
* You launch it with `<<<blocks, threads>>>`

Each kernel launch can create **thousands or millions of threads**.

---

### 3. Threads, Blocks, and Grids

These are the basic organizational units of CUDA execution:

| Level  | Description                                             | Example in your code |
| ------ | ------------------------------------------------------- | -------------------- |
| Thread | A single execution instance                             | `threadIdx.x = 0..4` |
| Block  | A group of threads that can cooperate via shared memory | `blockIdx.x = 0`     |
| Grid   | The full set of all blocks launched                     | `1 block` total      |

GPU hardware runs threads in groups called **warps** (usually 32 threads per warp) for efficient parallel execution.

---

### 4. Parallel Execution

Each GPU thread runs the same kernel code, but with different thread indices.
This enables **data parallelism** — multiple threads working on different parts of a problem simultaneously.

In this simple example, they all just print a line, but in real applications they could process elements of an image, compute matrix cells, etc.

---

### 5. Synchronization

Because CPU and GPU run asynchronously, you need `cudaDeviceSynchronize()` or other sync mechanisms to ensure proper timing or data consistency.

---

## 🧩 Summary

| Concept                   | Meaning in this code             |
| ------------------------- | -------------------------------- |
| `__global__`              | Marks a GPU kernel               |
| `<<<1,5>>>`               | Launches 1 block of 5 threads    |
| `threadIdx.x`             | Thread ID within the block       |
| `blockIdx.x`              | Block ID within the grid         |
| `cudaDeviceSynchronize()` | Waits for GPU to finish          |
| Parallelism               | 5 GPU threads run simultaneously |

---

Would you like me to extend this example next — for instance, to show how each thread can process a different array element (a real parallel computation example)?
