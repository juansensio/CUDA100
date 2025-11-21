# 1D Convolution

**Difficulty:** Easy

---

Write a function to perform 1D convolution with a "valid" boundary condition. Given two input arrays:

- `input`: a 1D array of 32-bit floating-point numbers
- `kernel`: a 1D array of 32-bit floating-point numbers (the filter)

compute the convolution result, storing it in the provided `output` array of size `input_size - kernel_size + 1`. The kernel should only be applied where it fully overlaps the input.

**Convolution Definition:**  
For each position `i` from `0` to `input_size - kernel_size`:
$$
output[i] = \sum_{j=0}^{kernel_size-1} input[i + j] * kernel[j]
$$


**Implementation Requirements:**
- Do **not** use external libraries
- The function `solve` must keep its signature unchanged
- Write the result into the `output` array

---

### Examples

**Example 1**  
Input:  

```
input = [1, 2, 3, 4, 5]  
kernel = [1, 0, -1]  
Output:  
[-2, -2, -2]
```

**Example 2**  
Input:  

```
input = [2, 4, 6, 8]  
kernel = [0.5, 0.2]  
Output:  
[1.8, 3.2, 4.6]
```

---

**Constraints**
- 1 ≤ input_size ≤ 1,500,000
- 1 ≤ kernel_size ≤ 2047
- kernel_size ≤ input_size