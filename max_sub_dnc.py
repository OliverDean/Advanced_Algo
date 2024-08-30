def max_crossing_sum(xs: list[int], left: int, mid: int, right: int) -> int:
    left_sum = float('-inf')
    total = 0
    for i in range(mid, left - 1, -1):
        total += xs[i]
        left_sum = max(left_sum, total)

    right_sum = float('-inf')
    total = 0
    for i in range(mid + 1, right + 1):
        total += xs[i]
        right_sum = max(right_sum, total)

    return left_sum + right_sum

def max_sum_subarray_divide_and_conquer(xs: list[int], left: int, right: int) -> int:
    """
    Finds the maximum sum of a subarray using a divide and conquer approach.

    The function has a worst-case time complexity of O(N log N).

    @param xs: The list of integers to analyze.
    @param left: The starting index of the current subarray.
    @param right: The ending index of the current subarray.
    @return: The maximum sum of any subarray within the given range.
    """
    if left == right:
        return max(0, xs[left])

    mid = (left + right) // 2
    left_max_sum = max_sum_subarray_divide_and_conquer(xs, left, mid)
    right_max_sum = max_sum_subarray_divide_and_conquer(xs, mid + 1, right)
    cross_max_sum = max_crossing_sum(xs, left, mid, right)

    return max(left_max_sum, right_max_sum, cross_max_sum)

def find_max_sum_subarray(xs: list[int]) -> int:
    """
    Wrapper function to find the maximum sum subarray in the entire list.
    
    @param xs: The list of integers to analyze.
    @return: The maximum sum of any subarray in the list.
    """
    if not xs:
        return 0
    return max_sum_subarray_divide_and_conquer(xs, 0, len(xs) - 1)
