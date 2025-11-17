# Understanding Memory

- **Global Memory:** Largest, high-latency, accessible by all threads and CPU, holds large datasets.
- **Shared Memory:** On-chip, fast, limited size, shared within thread block, useful for collaboration.
- **Registers (Local Memory):** Per-thread, fastest, very limited in size, used for local variables (spills to global if overused).

![memory](./pics/memory.png)

## global memory coalescing

threads are grouped in warps. memory accessess by threads in the same warp can be grouped and executed as one if:

- the threads are sequential in memory
- the threads are accessing consecutive memory locations

row-major!

## shared memory

...