import random
import time

def merge_and_count(array_list: list, temp_array: list, left_start: int, right_end: int) -> int:
    if left_start >= right_end:
        return 0

    mid_point = (left_start + right_end) // 2
    inversion_count = merge_and_count(array_list, temp_array, left_start, mid_point)
    inversion_count += merge_and_count(array_list, temp_array, mid_point + 1, right_end)
    inversion_count += merge_arrays(array_list, temp_array, left_start, mid_point, right_end)

    return inversion_count

def merge_arrays(array_list: list, temp_array: list, left_start: int, mid_point: int, right_end: int) -> int:
    left_index = left_start    
    right_index = mid_point + 1 
    merged_index = left_start    
    inversion_count = 0

    while left_index <= mid_point and right_index <= right_end:
        if array_list[left_index] <= array_list[right_index]:
            temp_array[merged_index] = array_list[left_index]
            left_index += 1
        else:
            temp_array[merged_index] = array_list[right_index]
            inversion_count += (mid_point - left_index + 1)
            right_index += 1
        merged_index += 1

    while left_index <= mid_point:
        temp_array[merged_index] = array_list[left_index]
        left_index += 1
        merged_index += 1

    while right_index <= right_end:
        temp_array[merged_index] = array_list[right_index]
        right_index += 1
        merged_index += 1

    for i in range(left_start, right_end + 1):
        array_list[i] = temp_array[i]

    return inversion_count

def count_inversions(array_list: list) -> int:
    """
    Counts the number of inversions in the list using an optimized
    merge sort algorithm.

    An inversion is a situation where a larger number precedes a smaller
    number in the list.

    @param array_list: The list of integers in which to count inversions.
    @return: The number of inversions in the list.
    """
    temp_array = [0] * len(array_list)
    return merge_and_count(array_list, temp_array, 0, len(array_list) - 1)

if __name__ == "__main__":

    N = 200000
    random_list = random.sample(range(1, N + 1), N)
    print("Randomly generated list of size:", N)

    start_time = time.time()

    inversion_count = count_inversions(random_list)

    end_time = time.time()

    print("Number of inversions:", inversion_count)
    print("Time taken:", end_time - start_time, "seconds")