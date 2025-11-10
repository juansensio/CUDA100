# Matrix Multiplication

- [matrix_multiply.cu](./matrix_multiply.cu): Naive implementation ~10x slower than PyTorch.
- [fast_matrix_multiply.cu](./fast_matrix_multiply.cu): Optimized implementation 

References:
- https://siboehm.com/articles/22/CUDA-MMM

## Problem Description

Write a program to multiply two matrices (\(A\) and \(B\)) of 32-bit floating-point numbers on a GPU. All matrices are stored in **row-major** order.

Given:
- Matrix **A** of dimensions \(M x K\)
- Matrix **B** of dimensions \(K x N\)
- Compute matrix **C** = **A** × **B**, resulting in dimensions \(M x N\).

---

###∫ Implementation Requirements

- Use **only native features** (no external libraries).
- Do **not** change the `solve` function signature.
- Write the result to matrix **C**.

---

### Example 1

**Input:**

- Matrix \(A\) (\(2 x 3\)):

  $$
  A = \begin{bmatrix}
  1.0 & 2.0 & 3.0 \\
  4.0 & 5.0 & 6.0
  \end{bmatrix}
  $$

- Matrix \(B\) (\(3 x 2\)):

  $$
  B = \begin{bmatrix}
  7.0 & 8.0 \\
  9.0 & 10.0 \\
  11.0 & 12.0
  \end{bmatrix}
  $$

**Output:**

- Matrix \(C\) (\(2 x 2\)):

  $$
  C = \begin{bmatrix}
  58.0 & 64.0 \\
  139.0 & 154.0
  \end{bmatrix}
  $$

---

### Example 2

**Input:**

- Matrix \(A\) (3 x 1):

  $$
  A = \begin{bmatrix}
  3.0 \\
  2.0 \\
  1.0
  \end{bmatrix}
  $$

- Matrix \(B\) (1 x 3):

  $$
  B = \begin{bmatrix}
  1.0 & 2.0 & 3.0
  \end{bmatrix}
  $$

**Output:**

- Matrix \(C\) (3 x 3):

  $$
  C = \begin{bmatrix}
  3.0 & 6.0 & 9.0 \\
  2.0 & 4.0 & 6.0 \\
  1.0 & 2.0 & 3.0
  \end{bmatrix}
  $$

---

### Constraints

- 1 <= M, N, K <= 8192
- Performance is measured at: \(M = 8192\), \(N = 6144\), \(K = 4096\)


## Optimizations

### Global memory coalescing

Threads are grouped in warps of 32 threads. Sequential memory accesses by threads that are part of the same warp can be grouped and executed as one -> *global memory coalescing*. We can achive simply by swapping the block and thread indices.