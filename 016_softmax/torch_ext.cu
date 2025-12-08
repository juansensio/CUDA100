#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include "main.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor softmax(torch::Tensor input) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    int N = input.numel();
    auto output = torch::empty({N}, input.options());

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / (threadsPerBlock);
    
    // Step 1: find max
    thrust::device_ptr<const float> dev_ptr_input(input.data_ptr<float>());
    float h_max = thrust::reduce(dev_ptr_input, dev_ptr_input + N, -INFINITY, thrust::maximum<float>());

    // step 2: compute exp(input - max) for numerical stability
    // probar ambas a ver que va mejor
    // thrust::device_ptr<float> dev_ptr_output(output_d);
    // thrust::transform(dev_ptr_output, dev_ptr_output + N, dev_ptr_output, [h_max](float x) { return exp(x - h_max); });
    exp_subtract_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input.data_ptr<float>(), output.data_ptr<float>(), h_max, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Step 3: Sum all exponentiated values using Thrust
    thrust::device_ptr<float> dev_ptr_output(output.data_ptr<float>());
    float h_sum = thrust::reduce(dev_ptr_output, dev_ptr_output + N, 0.0f, thrust::plus<float>());

    // Step 4: Normalize by dividing each element by sum
    normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(output.data_ptr<float>(), h_sum, N);    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return output;
}