def max_sum_subarray(N, A):
    """
    This function calculates the maximum sum of a contiguous subarray using Kadane's Algorithm.
    
    Problem Breakdown:
    - We are tasked with finding the maximum sum of any contiguous subarray in the given array `A`.
    - If the entire array contains negative numbers, we return 0 as the sum of the empty subarray is defined to be 0.
    
    Approach:
    1. Use **Kadane's Algorithm**:
       - Traverse through the array while maintaining two variables:
         - `current_sum`: the sum of the current subarray being evaluated.
         - `max_sum`: the maximum sum encountered so far.
       - If `current_sum` drops below 0, we reset it to 0, as a negative sum won't contribute positively to any future subarray.
       - At each step, update the `max_sum` if the `current_sum` is larger than the previously recorded `max_sum`.
    2. This algorithm runs in O(N) time, which is efficient for the given input size (up to 200,000 elements).

    Time Complexity: O(N) where N is the number of integers in the array.
    Space Complexity: O(1) as we only use a constant amount of extra space.

    Parameters:
    - N: The number of elements in the array A.
    - A: A list of integers representing the array.

    Returns:
    - The sum of the maximum sum subarray, or 0 if the array is empty or has no positive subarray.
    """
    
    max_sum = 0  # We start with the minimum valid sum (empty subarray)
    current_sum = 0

    for num in A:
        current_sum += num  # Add the current element to the running sum
        if current_sum > max_sum:
            max_sum = current_sum  # Update max_sum if the current_sum is larger
        if current_sum < 0:
            current_sum = 0  # Reset the current_sum if it drops below 0
    
    return max_sum

# Input
N = int(input())
A = list(map(int, input().split()))

# Output the maximum sum of a contiguous subarray
print(max_sum_subarray(N, A))
