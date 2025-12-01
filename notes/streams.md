# streams

Useful for performance optimisation in large systems.

it allows for overlapping computation and data transfer wehn launching multiple kernels.

copy1 -> kernel1 -> copy1 -> ...
        copy2 -> kernel2 -> copy2 -> ...

For example when training ML models, we can use streams to overlap the data transfer of batches from the CPU to the GPU with the actual computation of the model (forward and backward) on the GPU.

It can also be used to overlap multiple data transfers for the same kernel (for example two arrays required as inputs to the same kernel).