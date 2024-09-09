def max_sum_subarray(N, A):
    """
    This function finds the maximum sum of any contiguous subarray in the array A.

    Problem Breakdown:
    - We are tasked with finding the maximum sum of any contiguous subarray.
    - The sum of an empty subarray is defined as 0, which ensures the answer is always non-negative.
    
    Approach:
    1. Use a brute-force method, where we calculate the sum of every possible subarray and track the maximum sum encountered.
    2. Iterate over all possible starting and ending indices of the subarrays, calculate their sums, and keep track of the maximum sum.

    Time Complexity: O(N^2), where N is the number of elements in the array.
    Space Complexity: O(1), as we are only using a few variables to store intermediate results.

    Parameters:
    - N: The number of integers in the array.
    - A: A list of integers representing the array.

    Returns:
    - The sum of the maximum sum subarray.
    """
    
    max_sum = 0  # Initialize max_sum as 0 (for empty subarray)
    
    # Iterate over all possible subarrays
    for i in range(N):
        current_sum = 0
        for j in range(i, N):
            current_sum += A[j]  # Sum the subarray from i to j
            if current_sum > max_sum:
                max_sum = current_sum  # Update max_sum if we find a larger sum
    
    return max_sum

# Input
N = int(input())
A = list(map(int, input().split()))

# Output the maximum sum of a contiguous subarray
print(max_sum_subarray(N, A))
