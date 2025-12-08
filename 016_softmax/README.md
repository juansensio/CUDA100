# Problem: Parallel Softmax on GPU

Implement a CUDA program that computes the softmax of a 1D array of 32-bit floats. The softmax function for input array `input` of length `N` produces an array `output` of the same length, where:

```
output[i] = exp(input[i] - max(input)) / sum_j(exp(input[j] - max(input)))
```

**Numerical Stability:**  
To prevent overflow/underflow, use the "max trick": subtract the maximum value in `input` from each element before exponentiating.

---

## Requirements

- **No external libraries** are allowed; use only native/CUDA features.
- **Do not change** the provided `solve` function signature.
- The computed softmax values **must be written to the provided output array**.

---

## Example

**Example 1:**  
Input:  `[1.0, 2.0, 3.0]`, `N = 3`  
Output: `[0.090, 0.244, 0.665]` (approx)

**Example 2:**  
Input:  `[-10.0, -5.0, 0.0, 5.0, 10.0]`, `N = 5`  
Output: `[2.04e-09, 4.52e-07, 9.99e-06, 2.26e-02, 9.77e-01]` (approx)

---

## Constraints

- `1 ≤ N ≤ 500,000`

---