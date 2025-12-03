# Swish-Gated Linear Unit (SWiGLU)

## Problem Statement

Implement the forward pass of the Swish-Gated Linear Unit (SWiGLU) activation function for 1D input vectors.

- Given an input tensor of shape `[N]` (where $N$ is even), compute the output using the SWiGLU formula.
- Both the input and output tensors must be of type `float32`.

---

## SWiGLU Definition

Given an input vector $x$ of shape $[N]$ (with $N$ even):

1. **Split the input** into two halves:
   - $x_1 = x[:N/2]$
   - $x_2 = x[N/2:]$

2. **Compute SiLU activation** on the first half:
   - $\text{SiLU}(z) = z \cdot \sigma(z)$, where $\sigma(z) = \frac{1}{1 + e^{-z}}$

3. **Compute SWiGLU output (elementwise multiply the results):**
   - $\text{SWiGLU}(x) = \text{SiLU}(x_1) \times x_2$
   - The output shape is $[N/2]$

---

## Implementation Requirements

- **No external libraries** may be used—only native language features.
- The provided function signature (`solve` or equivalent) must not be changed.
- The computed result must be stored in the provided output tensor.

---

## Examples

**Example 1:**  
Input:  `[1.0, 2.0, 3.0, 4.0]`  ($N=4$)  
Output: `[2.1931758, 7.0463767]`

**Example 2:**  
Input: `[0.5, 1.0]`  ($N=2$)  
Output: `[0.31122968]`

---

## Constraints

- $1 \leq N \leq 100,\!000$
- $N$ is always even
- $-100.0 \leq$ input values $\leq 100.0$