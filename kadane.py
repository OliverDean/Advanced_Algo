def max_sum_subarray_kadane(xs: list[int]) -> int:
    """
    Finds the maximum sum of a subarray using Kadane's Algorithm.

    The function has a worst-case time complexity of O(N).

    @param xs: The list of integers to analyze.
    @return: The maximum sum of any subarray in the list.
    """
    max_current = 0
    max_global = 0

    for x in xs:
        max_current = max(x, max_current + x)
        max_global = max(max_global, max_current)

    return max_global
