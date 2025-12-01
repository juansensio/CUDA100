# Count Array Element

**Difficulty:** Easy

Write a GPU program to count how many times a specified integer value `k` appears in a given array of 32-bit integers.  
You are provided with:
- An input array `input` of length `N`.
- An integer value `k`.

The task is to compute the number of occurrences of `k` in the array and store the result in the `output` variable.

---

### Implementation Requirements
- Use only native GPU programming features (do **not** use external libraries).
- Do not modify the `solve` function signature.
- Store the final count in the provided `output` variable.

---

### Examples

**Example 1**  
Input: `input = [1, 2, 3, 4, 1]`, `k = 1`  
Output: `2`

**Example 2**  
Input: `input = [5, 10, 5, 2]`, `k = 11`  
Output: `0`

---

### Constraints
- `1 ≤ N ≤ 100,000,000`
- `1 ≤ input[i], k ≤ 100,000`
