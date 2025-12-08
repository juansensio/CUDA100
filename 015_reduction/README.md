# Problem: Parallel Reduction (Sum) on GPU

Implement a CUDA program to compute the sum of an array of 32-bit floats using parallel reduction on the GPU. The program should accept an input array and return the sum as a single float.

---

## Requirements

- **No external libraries** may be used; only GPU native/CUDA features are allowed.
- Do **not change** the provided `solve` function signature.
- The computed sum **must be written to the provided output variable**.

---

## Example

**Example 1:**  
Input:  `[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]`  
Output: `36.0`

**Example 2:**  
Input:  `[-2.5, 1.5, -1.0, 2.0]`  
Output: `0.0`

---

## Constraints

- $1 \leq N \leq 100,\!000,\!000$
- $-1000.0 \leq \text{input}[i] \leq 1000.0$
- The sum fits in a 32-bit float.

---