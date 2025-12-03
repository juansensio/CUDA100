# Sigmoid Linear Unit (SiLU)

**Difficulty**: Easy

---

## Problem Description

Implement the SiLU (Sigmoid Linear Unit) activation function forward pass for 1D input vectors.

Given:
- An input tensor ("input") of shape `[N]`, where `N` is the number of elements.
- Compute the output tensor ("output") of the same shape using the elementwise formula:

### SiLU Formula

For each element `x` in the input:
```
SiLU(x) = x * sigmoid(x)
        = x * (1 / (1 + exp(-x)))
```

---

## Implementation Requirements

- Use only native language features (no external libraries).
- The function signature `solve(input, output)` must remain unchanged.
- Store the final computed result in the `output` tensor.

---

## Examples

**Example 1**  
Input:  `input = [0.5, 1.0, -0.5]`  
Output: `output = [0.3112295, 0.731059, -0.1887705]`

**Example 2**  
Input:  `input = [-1.0, -2.0, -3.0, -4.0, -5.0]`  
Output: `output = [-0.26894143, -0.23840584, -0.14227763, -0.07194484, -0.03346425]`

---

## Constraints

- `1 ≤ N ≤ 10,000`
- Each input value: `-100.0 ≤ input[i] ≤ 100.0`