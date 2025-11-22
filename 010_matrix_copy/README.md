# Matrix Copy

**Difficulty:** Easy

---

Implement a CUDA program that copies an $N \times N$ matrix of 32-bit floating point numbers from input array **A** to output array **B**. The operation should be a direct, element-wise copy, so that $B[i][j] = A[i][j]$ for all valid indices $i, j$.

## Implementation Requirements

- Do **not** use external libraries.
- Do **not** change the signature of the `solve` function.
- The result must be written to matrix **B**.

## Examples

**Example 1**  
Input:  
```
A = [[1.0, 2.0],
     [3.0, 4.0]]
```
Output:  
```
B = [[1.0, 2.0],
     [3.0, 4.0]]
```

**Example 2**  
Input:  
```
A = [[5.5, 6.6, 7.7],
     [8.8, 9.9, 10.1],
     [11.2, 12.3, 13.4]]
```
Output:  
```
B = [[5.5, 6.6, 7.7],
     [8.8, 9.9, 10.1],
     [11.2, 12.3, 13.4]]
```

## Constraints

- $1 \leq N \leq 4096$
- All elements are 32-bit floating point numbers