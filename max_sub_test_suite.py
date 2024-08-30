import random
import time

def max_sum_subarray_naive(xs: list[int]) -> int:
    best_sum = 0
    for lwr in range(len(xs)):
        for upr in range(lwr + 1, len(xs) + 1):
            total_sum = 0
            for i in range(lwr, upr):
                total_sum += xs[i]
            best_sum = max(best_sum, total_sum)
    return best_sum

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
    if left == right:
        return max(0, xs[left])

    mid = (left + right) // 2
    left_max_sum = max_sum_subarray_divide_and_conquer(xs, left, mid)
    right_max_sum = max_sum_subarray_divide_and_conquer(xs, mid + 1, right)
    cross_max_sum = max_crossing_sum(xs, left, mid, right)

    return max(left_max_sum, right_max_sum, cross_max_sum)

def find_max_sum_subarray(xs: list[int]) -> int:
    if not xs:
        return 0
    return max_sum_subarray_divide_and_conquer(xs, 0, len(xs) - 1)

def max_sum_subarray_kadane(xs: list[int]) -> int:
    max_current = 0
    max_global = 0

    for x in xs:
        max_current = max(x, max_current + x)
        max_global = max(max_global, max_current)

    return max_global

if __name__ == "__main__":
    # Generate a random list of integers
    N = 1000  # Change this value as needed for testing
    random_list = random.choices(range(-100, 100), k=N)

    print(f"Testing on a list of size {N}:\n")

    # Test naive O(N^3) solution
    start_time = time.time()
    result_naive = max_sum_subarray_naive(random_list)
    naive_time = time.time() - start_time
    print(f"Naive O(N^3) result: {result_naive}, Time: {naive_time:.6f} seconds")

    # Test divide and conquer O(N log N) solution
    start_time = time.time()
    result_divide_conquer = find_max_sum_subarray(random_list)
    divide_conquer_time = time.time() - start_time
    print(f"Divide and Conquer O(N log N) result: {result_divide_conquer}, Time: {divide_conquer_time:.6f} seconds")

    # Test Kadane's O(N) solution
    start_time = time.time()
    result_kadane = max_sum_subarray_kadane(random_list)
    kadane_time = time.time() - start_time
    print(f"Kadane's O(N) result: {result_kadane}, Time: {kadane_time:.6f} seconds")
