# Color Inversion

**Difficulty:** Easy

Write a program to invert the colors of an image. The image is represented as a 1D array of RGBA (Red, Green, Blue, Alpha) values, where each component is an 8-bit unsigned integer (`unsigned char`).

**Color inversion** is performed by subtracting each color component (R, G, B) from 255. The Alpha component should remain unchanged.

If a pixel's RGBA value is \([R, G, B, A]\), the inverted pixel is \([255-R, 255-G, 255-B, A]\).

---

Let:

- $\text{image}[i]$ be the flat RGBA array,
- $\text{width}$ = number of columns,
- $\text{height}$ = number of rows,
- Total elements: $N = \text{width} \times \text{height} \times 4$.

The algorithm:

For $i$ in $0, 4, 8, \ldots, N-4$:

$$
\begin{aligned}
\text{image}[i + 0] &:= 255 - \text{image}[i + 0] \quad &\text{(R)} \\
\text{image}[i + 1] &:= 255 - \text{image}[i + 1] \quad &\text{(G)} \\
\text{image}[i + 2] &:= 255 - \text{image}[i + 2] \quad &\text{(B)} \\
\text{image}[i + 3] &:= \text{image}[i + 3]           \quad &\text{(A, unchanged)}
\end{aligned}
$$

---

## Implementation Requirements

- Use only native features (no external libraries).
- The `solve` function signature must remain unchanged.
- The result must be stored in the input array `image`.

---

## Examples

**Example 1:**  
Input:  
$$
\text{image} = [255, 0, 128, 255,\, 0, 255, 0, 255]
$$  
$\text{width}=1,\, \text{height}=2$

Output:  
$$
[0, 255, 127, 255,\, 255, 0, 255, 255]
$$

**Example 2:**  
Input:  
$$
\text{image} = [10, 20, 30, 255,\, 100, 150, 200, 255]
$$  
$\text{width}=2,\, \text{height}=1$

Output:  
$$
[245, 235, 225, 255,\, 155, 105, 55, 255]
$$

---

## Constraints

$$
\begin{aligned}
1 &\leq \text{width} \leq 4096 \\
1 &\leq \text{height} \leq 4096 \\
\text{width} \times \text{height} &\leq 8{,}388{,}608
\end{aligned}
$$