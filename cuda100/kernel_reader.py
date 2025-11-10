def kernel_reader(file):
    # Read the CUDA file and extract the kernel
    with open(file, 'r') as f:
        content = f.read()

    # Find the start of the kernel (__global__)
    start_idx = content.find('__global__')
    if start_idx == -1:
        raise ValueError("Kernel not found")

    # Find the matching closing brace for the kernel function
    # We need to count braces to find the correct closing brace
    brace_count = 0
    start_brace = content.find('{', start_idx)
    if start_brace == -1:
        raise ValueError("Opening brace not found")

    end_idx = start_brace
    for i in range(start_brace, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break

    # Extract the kernel string
    return content[start_idx:end_idx]