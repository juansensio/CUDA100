# Understanding Memory

## global memory coalescing

threads are grouped in warps. memory accessess by threads in the same warp can be grouped and executed as one if:

- the threads are sequential in memory
- the threads are accessing consecutive memory locations

row-major!

## shared memory

...