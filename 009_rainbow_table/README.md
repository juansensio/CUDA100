# Rainbow Table

**Difficulty:** Easy

---

Implement a program that performs **R** rounds of parallel hashing on an array of 32-bit integers using the provided hash function. The hash function should be applied iteratively: the output of each round becomes the input of the next round.

## Implementation Requirements

- Do **not** use external libraries.
- You must not change the signature of the `solve` function.
- The final result must be written to the `output` array.

## Examples

**Example 1**  
Input:  
```
numbers = [123, 456, 789]
R = 2
```
Output:  
```
hashes = [1636807824, 1273011621, 2193987222]
```

**Example 2**  
Input:  
```
numbers = [0, 1, 2147483647]
R = 3
```
Output:  
```
hashes = [96754810, 3571711400, 2006156166]
```

## Constraints

- 1 ≤ N ≤ 10,000,000
- 1 ≤ R ≤ 100
- 0 ≤ input[i] ≤ 2147483647