# Softmax Attention

**Difficulty:** Medium

## Problem Statement

Implement a GPU program that computes the softmax attention operation for a set of input matrices. Specifically, for given:

- Query matrix **Q** of size (M × d)
- Key matrix **K** of size (N × d)
- Value matrix **V** of size (N × d)

Compute the attention output matrix with the following formula:

$$
\mathrm{Attention}(\mathbf{Q},\,\mathbf{K},\,\mathbf{V}) = \mathrm{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}
$$

The softmax is computed for each query row (**row-wise**) over its dot products with keys.



## Implementation Requirements

- Only use **GPU native features** (do **not** use external libraries)
- The provided `solve` function signature **must not be changed**
- Write results to the output matrix `output`

---

## Example 1

**Input:**

```
Q (2×4):
[1, 0, 0, 0]
[0, 1, 0, 0]

K (3×4):
[1, 0, 0, 0]
[0, 1, 0, 0]
[0, 0, 1, 0]

V (3×4):
[1, 2, 3, 4]
[5, 6, 7, 8]
[9, 10, 11, 12]
```

**Output:**

```
output (2×4):
[4.29, 5.29, 6.29, 7.29]
[5, 6, 7, 8]
```

---

## Example 2

**Input:**

```
Q (1×2):
[1, 2]

K (2×2):
[1, 0]
[0, 1]

V (2×2):
[3, 4]
[5, 6]
```

**Output:**

```
output (1×2):
[4.34, 5.34]
```

---

## Constraints

- Q: (M × d), K: (N × d), V: (N × d)
- 1 ≤ M, N ≤ 100,000
- 1 ≤ d ≤ 1024