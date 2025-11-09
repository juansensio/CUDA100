# Vector Addition

Write a program that performs element-wise addition of two vectors (`A` and `B`) containing 32-bit floating point numbers, utilizing the GPU. The two input vectors must have the same length, and the output should be a single vector (`C`) where each element is the sum of the corresponding elements in `A` and `B`.

**Requirements:**
- Do not use any external libraries.
- The `solve` function signature must not be changed.
- Store the final result in vector `C`.

**Examples:**

- *Example 1*  
  Input:  
  &nbsp;&nbsp;A = [1.0, 2.0, 3.0, 4.0]  
  &nbsp;&nbsp;B = [5.0, 6.0, 7.0, 8.0]  
  Output:  
  &nbsp;&nbsp;C = [6.0, 8.0, 10.0, 12.0]

- *Example 2*  
  Input:  
  &nbsp;&nbsp;A = [1.5, 1.5, 1.5]  
  &nbsp;&nbsp;B = [2.3, 2.3, 2.3]  
  Output:  
  &nbsp;&nbsp;C = [3.8, 3.8, 3.8]

**Constraints:**
- The input vectors `A` and `B` must have the same length.
- $1 \leq N \leq 100,\!000,\!000$, where $N$ is the length of `A` and `B`.