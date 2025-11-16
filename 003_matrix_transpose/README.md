# Matrix Transpose

**Difficulty:** Easy

Write a program that transposes a matrix of 32-bit floating point numbers on a GPU. The transpose of a matrix switches its rows and columns. Given a matrix $A$ of dimensions $r \times c$, the transpose $A^T$ will have dimensions $c \times r$. All matrices are stored in row-major format.

## Examples

### Example 1

Input: $2 \times 3$ matrix
$$
A = \begin{bmatrix}
a_{00} & a_{01} & a_{02} \\
a_{10} & a_{11} & a_{12}
\end{bmatrix}
$$

Output: $3 \times 2$ matrix
$$
A^\top = \begin{bmatrix}
a_{00} & a_{10} \\
a_{01} & a_{11} \\
a_{02} & a_{12}
\end{bmatrix}
$$

---

### Example 2

Input: $3 \times 1$ matrix
$$
A = \begin{bmatrix}
a_{00} \\
a_{10} \\
a_{20}
\end{bmatrix}
$$

Output: $1 \times 3$ matrix
$$
A^\top = \begin{bmatrix}
a_{00} & a_{10} & a_{20}
\end{bmatrix}
$$

## Constraints

- $1 \leq r, c \leq 8192$
- Input matrix dimensions: $r \times c$
- Output matrix dimensions: $c \times r$