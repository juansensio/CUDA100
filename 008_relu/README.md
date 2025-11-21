# ReLU

**Difficulty:** Easy

Implement a program to compute the Rectified Linear Unit (ReLU) activation on a vector of 32-bit floating point numbers. The ReLU function replaces each negative value in the input with 0, leaving non-negative values unchanged.

---

## Implementation Requirements

- Do **not** use any external libraries.
- Do **not** change the signature of the `solve` function.
- The final result must be stored in the `output` array.

---

## Examples

**Example 1**  
Input:  
`input = [-2.0, -1.0, 0.0, 1.0, 2.0]`  
Output:  
`output = [0.0, 0.0, 0.0, 1.0, 2.0]`

**Example 2**  
Input:  
`input = [-3.5, 0.0, 4.2]`  
Output:  
`output = [0.0, 0.0, 4.2]`

---

## Constraints

- `1 ≤ N ≤ 100,000,000`
