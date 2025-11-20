# Matrix Addition

**Difficulty:** Easy

Write a program to perform element-wise addition of two matrices of 32-bit floating point numbers on a GPU. The program accepts two input matrices of the same shape and outputs a matrix containing their element-wise sum.

## Requirements

- **No external libraries** allowed.
- **Do not modify** the provided `solve` function signature.
- Store the resulting matrix in `C`.

## Examples

**Example 1**

Input:
```
A = [[1.0, 2.0],
     [3.0, 4.0]]

B = [[5.0, 6.0],
     [7.0, 8.0]]
```
Output:
```
C = [[6.0, 8.0],
     [10.0, 12.0]]
```

**Example 2**

Input:
```
A = [[1.5, 2.5, 3.5],
     [4.5, 5.5, 6.5],
     [7.5, 8.5, 9.5]]

B = [[0.5, 0.5, 0.5],
     [0.5, 0.5, 0.5],
     [0.5, 0.5, 0.5]]
```
Output:
```
C = [[2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0],
     [8.0, 9.0, 10.0]]
```

## Constraints

- The input matrices `A` and `B` must have the same dimensions.
- Dimensions: `1 ≤ N ≤ 4096`
- All elements are 32-bit floating-point numbers.