"""
Question 1: Total Distance Walked by Friends

We are asked to determine the complexity of sorting a list of friends by height, and the total distance walked by each person. Sorting the list minimizes the total distance.


Question Explanation: Group Photo Problem (Distance Walked)

**Problem**:
You and your N friends are standing in a line and need to rearrange yourselves in increasing order of height (shortest on the left, tallest on the right). Each person will walk a certain distance to achieve this, and we need to determine the time complexity for calculating the total distance walked by all participants.

**Key Concept**:
The total distance walked, D, is proportional to the number of **inversions** in the original sequence of heights. An inversion occurs when a person who is taller stands in front of someone who is shorter (i.e., the sequence is "inverted"). In the worst case, where the sequence is in reverse order (i.e., all people need to swap places), the total number of inversions is **O(N^2)**.

**Correct Answer**:
- **O(N^2)**: This corresponds to the case when the sequence is exactly the reverse of the sorted order. To rearrange everyone into the correct order, each person must potentially walk past every other person, leading to a quadratic number of swaps or movements.

**Incorrect Answers**:
- **O(N)**: This complexity is too low for this problem, as sorting the list and counting inversions both require more than linear time.
- **O(N lg N)**: This complexity is associated with sorting algorithms like Merge Sort, but it doesn't account for the walking distance (inversions).
- **O(N^2 lg N)**: This complexity is not necessary; the correct complexity for counting inversions (and therefore the total distance walked) is simply O(N^2).

**Response Feedback**:
- The total distance walked is proportional to the number of inversions, which is at worst O(N^2). This occurs when the sequence is in reverse order, as each person must walk past every other person, leading to quadratic complexity.



----------------------------------------

Question 2: Time Complexity of Merge Sort

Merge Sort is a divide-and-conquer algorithm. It recursively splits the list into two halves, sorts each half, and then merges the sorted halves together. Merging two sorted lists takes linear time (O(N)) per merge, and there are O(log N) levels of recursion due to halving the list.

The valid argument for the time complexity of Merge Sort is:

- **Correct Answer**: 
  - "Merge sort is a divide and conquer algorithm that splits a list of length N into two lists of length N/2, recursively sorts these sublists, and then merges them together. Two sorted lists can be merged in linear time by repeatedly taking the smallest remaining element from either list. This means that each time an element is involved in a merge contributes O(1) time. Since the length of the lists halves with each recurrence, and a list of length < 2 is already sorted, the maximum recursion depth is O(log N). This is also therefore a limit on the number of times each element can be involved in a merge. Across all N elements this gives a total complexity of O(N log N)."

----------------------------------------

Question 3: Longest Increasing Subsequence (LIS) Using Dynamic Programming

**Problem**: Given a sequence of integers, find the length of the longest subsequence that is strictly increasing. This can be solved using a dynamic programming (DP) approach.

**State Definition**:
- Let `dp[i]` represent the length of the longest increasing subsequence that ends at index `i`.

**Recurrence Relation**:
- For each `i` from 0 to N-1, check all previous elements `j` (where `0 ≤ j < i`). If `S[i] > S[j]`, update `dp[i] = max(dp[i], dp[j] + 1)`.

**Base Case**:
- For each index `i`, the subsequence that includes only `S[i]` has a length of 1, so `dp[i] = 1`.

**Algorithm**:
1. Initialize `dp[i] = 1` for all `i`.
2. For each element `S[i]` in the sequence, loop over all `S[j]` for `j < i`. If `S[i] > S[j]`, update `dp[i] = max(dp[i], dp[j] + 1)`.
3. The result is the maximum value in the `dp` array.

**Time Complexity**:
- **O(N^2)**: For each `i`, we loop over all `j < i`, resulting in O(N^2) comparisons.

**Space Complexity**:
- **O(N)**: We only need an array `dp` of size `N`.

Example Python Code:
```python
def longest_increasing_subsequence(S):
    N = len(S)
    dp = [1] * N

    for i in range(1, N):
        for j in range(i):
            if S[i] > S[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

# Example usage:
S = [2, 1, 1, 4, 1, 9, 5, 7]
print(longest_increasing_subsequence(S))  # Output: 4 (LIS could be [1, 4, 5, 7])

"""