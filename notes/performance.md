# Performance Metrics

Being able to measure the performance of a CUDA program is crucial for optimizing it. There are several metrics that can be used to measure the performance of a CUDA program.

> Also important to check correctness of the program, with test cases or comparing with other implementations (e.g. PyTorch, cuBLAS, etc.).

## Execution Time

The simplest metric is the execution time of the program. This is the time it takes for the program to complete its execution. The faster the program executes, the better. 

## FLOPs

FLOPs stands for "Floating Point Operations per Second." It's a measure of a computer's performance, especially in tasks that require heavy numerical calculations, like scientific computing, AI, and graphics. In the context of CUDA and GPU programming, we frequently use FLOPs to quantify:

- **Theoretical peak performance:** The maximum number of floating point operations a GPU can perform per second, based on hardware specs.
- **Actual achieved performance:** Calculated by counting the total number of floating point operations your program performs and dividing by the elapsed time (in seconds).

This metric depends on the hardware and algorithm used.

For a matrix multiplication of two MxK and KxN matrices, the total floating point operations is usually counted as 2xMxNxK (since each multiply-add counts as two operations). FLOPs is then calculated as the total number of floating point operations divided by the execution time.

Our goal is to maximize the FLOPs of the program, ideally close to the theoretical peak performance of the hardware.

### 3090 RTX

- FP16 (half) 35.58 TFLOPS (1:1)
- FP32 (float) 35.58 TFLOPS
- FP64 (double) 556.0 GFLOPS (1:64)
- Bandwidth 936.2 GB/s

## Other Metrics

- Memory Bandwidth: The amount of data transferred between the CPU and GPU per second.
- GPU Utilization: The percentage of time the GPU is busy executing tasks.
- GPU Power Usage: The amount of power the GPU is consuming.