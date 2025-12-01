# Count 2D Array Element

**Difficulty:** Easy

Write a GPU program that counts the number of elements equal to a given integer `k` in a 2D array of 32-bit integers.  
Given an input 2D array `input` of size `N x M` and an integer `k`, compute how many elements in the array have the value `k`.

---

## Requirements

- Use only native features (no external libraries permitted)
- You must keep the `solve` function signature unchanged
- Store the final count in the variable `output`

---

## Example 1

**Input:**
```
input = [
  [1, 2, 3],
  [4, 5, 1]
]
k = 1
```
**Output:**
```
output = 2
```

## Example 2

**Input:**
```
input = [
  [5, 10],
  [5, 2]
]
k = 1
```
**Output:**
```
output = 0
```

---

## Constraints

- 1 ≤ N, M ≤ 10,000
- 1 ≤ input[i][j], k ≤ 100