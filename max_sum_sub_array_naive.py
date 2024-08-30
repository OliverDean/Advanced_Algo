def max_sum_subarray_naive(xs: list[int]) -> int:
    """
    Finds the maximum sum of a subarray in the given list using a naive O(N^3) approach.

    A subarray is defined as a contiguous subsequence of elements within the list. 
    The function returns the sum of the subarray with the largest sum.

    @param xs: The list of integers to analyze.
    @return: The maximum sum of any subarray in the list.
    """
    best_sum = 0
    for lwr in range(len(xs)):
        for upr in range(lwr + 1, len(xs) + 1):
            total_sum = 0
            for i in range(lwr, upr):
                total_sum += xs[i]
            best_sum = max(best_sum, total_sum)
    return best_sum
