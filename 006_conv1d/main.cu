#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cudnn.h>

#define CUDA_OK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDNN_OK(call) do { \
    cudnnStatus_t status = (call); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error %s at %s:%d\n", cudnnGetErrorString(status), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void convolution_1d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ kernel, 
    float* __restrict__ output,
    int input_size, int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size - kernel_size + 1) {
        float sum = 0.0f;
        for (int i = 0; i < kernel_size; ++i) {
            sum += input[idx + i] * kernel[i];
        }
        output[idx] = sum;
    }
    // ~2 TFLOPs
}

// Calculate grid and block dimensions for convolution kernel
void get_launch_config(int output_size, dim3* threadsPerBlock, dim3* blocksPerGrid) {
    *threadsPerBlock = dim3(256, 1, 1);
    *blocksPerGrid = dim3((output_size + threadsPerBlock->x - 1) / threadsPerBlock->x, 1, 1);
}

void conv1d_custom(const float* h_input, const float* h_kernel, float* h_output,
                   int input_size, int kernel_size, int output_size) {
    float *d_input, *d_kernel, *d_output;
    CUDA_OK(cudaMalloc((void**)&d_input, input_size * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&d_kernel, kernel_size * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&d_output, output_size * sizeof(float)));
    
    CUDA_OK(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 threadsPerBlock, blocksPerGrid;
    get_launch_config(output_size, &threadsPerBlock, &blocksPerGrid);
    
    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, input_size, kernel_size);
    
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_OK(cudaFree(d_input));
    CUDA_OK(cudaFree(d_kernel));
    CUDA_OK(cudaFree(d_output));
}

void conv1d_cudnn(const float* h_input, const float* h_kernel, float* h_output,
                  int input_size, int kernel_size, int output_size,
                  cudnnHandle_t cudnn) {
    // Create tensor descriptors
    int in_n = 1, in_c = 1, in_h = 1, in_w = input_size;
    int in_dims[4] = {in_n, in_c, in_h, in_w};
    int in_strides[4] = {in_c * in_h * in_w, in_h * in_w, in_w, 1};

    cudnnTensorDescriptor_t xDesc;
    CUDNN_OK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_OK(cudnnSetTensorNdDescriptor(xDesc, CUDNN_DATA_FLOAT, 4, in_dims, in_strides));

    int out_c = 1;
    int filt_dims[4] = {out_c, in_c, 1, kernel_size};

    cudnnFilterDescriptor_t wDesc;
    CUDNN_OK(cudnnCreateFilterDescriptor(&wDesc));
    CUDNN_OK(cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filt_dims));

    cudnnConvolutionDescriptor_t convDesc;
    CUDNN_OK(cudnnCreateConvolutionDescriptor(&convDesc));

    int padA[2] = {0, 0};
    int strideA[2] = {1, 1};
    int dilationA[2] = {1, 1};

    CUDNN_OK(cudnnSetConvolutionNdDescriptor(convDesc, 2, padA, strideA, dilationA,
                                             CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    int out_dims[4];
    CUDNN_OK(cudnnGetConvolutionNdForwardOutputDim(convDesc, xDesc, wDesc, 4, out_dims));

    int out_strides[4] = {out_c * out_dims[2] * out_dims[3], out_dims[2] * out_dims[3], out_dims[3], 1};

    cudnnTensorDescriptor_t yDesc;
    CUDNN_OK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_OK(cudnnSetTensorNdDescriptor(yDesc, CUDNN_DATA_FLOAT, 4, out_dims, out_strides));

    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    CUDA_OK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_kernel, kernel_size * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_output, output_size * sizeof(float)));

    CUDA_OK(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    // Select algorithm
    cudnnConvolutionFwdAlgo_t algo;
    int requestedAlgoCount = 8;
    int returnedAlgoCount = 0;
    cudnnConvolutionFwdAlgoPerf_t algoPerf[8];
    
    CUDNN_OK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, xDesc, wDesc, convDesc, yDesc,
                                                     requestedAlgoCount, &returnedAlgoCount, algoPerf));
    
    if (returnedAlgoCount == 0) {
        fprintf(stderr, "No valid convolution algorithm found\n");
        exit(EXIT_FAILURE);
    }
    
    algo = algoPerf[0].algo;

    size_t workspace_bytes = 0;
    CUDNN_OK(cudnnGetConvolutionForwardWorkspaceSize(cudnn, xDesc, wDesc, convDesc, yDesc,
                                                      algo, &workspace_bytes));

    void* d_workspace = nullptr;
    if (workspace_bytes > 0) {
        CUDA_OK(cudaMalloc(&d_workspace, workspace_bytes));
    }

    // Run convolution
    float alpha = 1.0f;
    float beta = 0.0f;

    CUDNN_OK(cudnnConvolutionForward(cudnn, &alpha, xDesc, d_input, wDesc, d_kernel, convDesc,
                                     algo, d_workspace, workspace_bytes, &beta, yDesc, d_output));

    CUDA_OK(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    if (d_workspace) cudaFree(d_workspace);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
}

void benchmark() {
    printf("\nBenchmarking Custom CUDA kernel vs cuDNN for various sizes...\n");
    printf("%-10s %-8s %-18s %-18s %-18s %-18s %-18s\n", 
           "N", "K", "Custom (ms)", "TFLOPs", "cuDNN (ms)", "TFLOPs", "Speedup");
    
    int Ns[] = {10000, 100000, 1500000};
    int kernel_sizes[] = {512, 1024, 2048};
    const int num_sizes = sizeof(Ns) / sizeof(Ns[0]);
    const int num_kernels = sizeof(kernel_sizes) / sizeof(kernel_sizes[0]);
    const int num_runs = 10;
    
    // Create cuDNN handle once for reuse
    cudnnHandle_t cudnn;
    CUDNN_OK(cudnnCreate(&cudnn));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int size_idx = 0; size_idx < num_sizes; ++size_idx) {
        for (int kernel_idx = 0; kernel_idx < num_kernels; ++kernel_idx) {
            int input_size = Ns[size_idx];
            int kernel_size = kernel_sizes[kernel_idx];
            
            if (kernel_size > input_size) {
                continue;  // Skip invalid configurations
            }
            
            int output_size = input_size - kernel_size + 1;
            
            // Allocate host memory
            float *h_input = (float*)malloc(input_size * sizeof(float));
            float *h_kernel = (float*)malloc(kernel_size * sizeof(float));
            float *h_output_custom = (float*)malloc(output_size * sizeof(float));
            float *h_output_cudnn = (float*)malloc(output_size * sizeof(float));
            
            if (!h_input || !h_kernel || !h_output_custom || !h_output_cudnn) {
                fprintf(stderr, "Host malloc failed\n");
                exit(EXIT_FAILURE);
            }
            
            // Fill with random data
            for (int i = 0; i < input_size; ++i) {
                h_input[i] = (float)(rand()) / (float)(RAND_MAX);
            }
            for (int i = 0; i < kernel_size; ++i) {
                h_kernel[i] = (float)(rand()) / (float)(RAND_MAX);
            }
            
            // Warm-up runs
            conv1d_custom(h_input, h_kernel, h_output_custom, input_size, kernel_size, output_size);
            conv1d_cudnn(h_input, h_kernel, h_output_cudnn, input_size, kernel_size, output_size, cudnn);
            CUDA_OK(cudaDeviceSynchronize());
            
            // Benchmark custom kernel
            float ms_custom = 0.0f;
            for (int run = 0; run < num_runs; ++run) {
                float *d_input, *d_kernel, *d_output;
                CUDA_OK(cudaMalloc((void**)&d_input, input_size * sizeof(float)));
                CUDA_OK(cudaMalloc((void**)&d_kernel, kernel_size * sizeof(float)));
                CUDA_OK(cudaMalloc((void**)&d_output, output_size * sizeof(float)));
                
                CUDA_OK(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_OK(cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));
                
                dim3 threadsPerBlock, blocksPerGrid;
                get_launch_config(output_size, &threadsPerBlock, &blocksPerGrid);
                
                cudaEventRecord(start);
                convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, input_size, kernel_size);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                
                float t;
                cudaEventElapsedTime(&t, start, stop);
                ms_custom += t;
                
                CUDA_OK(cudaFree(d_input));
                CUDA_OK(cudaFree(d_kernel));
                CUDA_OK(cudaFree(d_output));
            }
            ms_custom /= num_runs;
            
            // Benchmark cuDNN
            float ms_cudnn = 0.0f;
            for (int run = 0; run < num_runs; ++run) {
                cudaEventRecord(start);
                conv1d_cudnn(h_input, h_kernel, h_output_cudnn, input_size, kernel_size, output_size, cudnn);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                
                float t;
                cudaEventElapsedTime(&t, start, stop);
                ms_cudnn += t;
            }
            ms_cudnn /= num_runs;
            
            // Compute FLOPs: conv1d for each output element is kernel_size multiplications and (kernel_size - 1) adds
            // So 2*kernel_size - 1 FLOPs per output element
            long long flops_per_output = 2LL * kernel_size - 1;
            long long total_flops = (long long)output_size * flops_per_output;
            
            // TFLOPs = total_flops / (time_in_ms * 1e-3) / 1e12 = total_flops / (time_in_ms * 1e9)
            double custom_tflops = (ms_custom > 0.0) ? ((double)total_flops / (ms_custom * 1e9)) : 0.0;
            double cudnn_tflops = (ms_cudnn > 0.0) ? ((double)total_flops / (ms_cudnn * 1e9)) : 0.0;
            
            // Speedup: cuDNN time / Custom time (how much faster cuDNN is)
            double speedup = (ms_custom > 0.0) ? (ms_cudnn / ms_custom) : 0.0;
            
            printf("%-10d %-8d %-18.4f %-18.6f %-18.4f %-18.6f %-18.4f\n",
                   input_size, kernel_size, ms_custom, custom_tflops, ms_cudnn, cudnn_tflops, speedup);
            
            free(h_input);
            free(h_kernel);
            free(h_output_custom);
            free(h_output_cudnn);
        }
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudnnDestroy(cudnn);
}

int main() {
    printf("Testing 1D convolution...\n");
    float input[]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float kernel[]  = {1.0f, 0.0f, -1.0f};
    float expected[] = {-2.0f, -2.0f, -2.0f};
    int input_size = 5;
    int kernel_size = 3;
    int output_size = input_size - kernel_size + 1;
    float output[output_size];
    
    float* input_d, *kernel_d, *output_d;
    CUDA_OK(cudaMalloc((void**)&input_d, input_size * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&kernel_d, kernel_size * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&output_d, output_size * sizeof(float)));
    CUDA_OK(cudaMemcpy(input_d, input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(kernel_d, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 threadsPerBlock, blocksPerGrid;
    get_launch_config(output_size, &threadsPerBlock, &blocksPerGrid);

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_d, kernel_d, output_d, input_size, kernel_size);

    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(output, output_d, output_size * sizeof(float), cudaMemcpyDeviceToHost));
        
    // Verify the result
    for (int i = 0; i < output_size; ++i) { 
        if (output[i] != expected[i]) {
            printf("Error at index %d: expected %.1f, got %.1f\n", i, expected[i], output[i]);
            CUDA_OK(cudaFree(input_d));
            CUDA_OK(cudaFree(kernel_d));
            CUDA_OK(cudaFree(output_d));
            return EXIT_FAILURE;
        }
    }
    printf("✓ Result is correct!\n");
    CUDA_OK(cudaFree(input_d));
    CUDA_OK(cudaFree(kernel_d));
    CUDA_OK(cudaFree(output_d));

    // testing cudnn
    // Same test data as your CUDA 1D conv
    float h_input[]    = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float h_kernel[]   = {1.0f, 0.0f, -1.0f};
    float h_expected[] = {-2.0f, -2.0f, -2.0f};
    float h_output[output_size];

    // cuDNN handle
    cudnnHandle_t cudnn;
    CUDNN_OK(cudnnCreate(&cudnn));

    // Use the helper function
    conv1d_cudnn(h_input, h_kernel, h_output, input_size, kernel_size, output_size, cudnn);

    // Verify result
    bool ok = true;
    for (int i = 0; i < output_size; ++i) {
        if (fabsf(h_output[i] - h_expected[i]) > 1e-5f) {
            printf("Error at index %d: expected %.1f, got %.5f\n",
                   i, h_expected[i], h_output[i]);
            ok = false;
        }
    }

    if (ok) {
        printf("✓ cuDNN result is correct!\n");
    } else {
        printf("✗ cuDNN result is incorrect.\n");
    }

    cudnnDestroy(cudnn);

    benchmark();
    return 0;
}
