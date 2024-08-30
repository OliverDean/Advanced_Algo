def insertion_sort(xs: list) -> int:
    """
    Performs an insertion sort on the input list and returns the number of inversions.

    An inversion is a situation where a larger number precedes a smaller number in the list.

    @param xs: The list of integers to be sorted.
    @return: The number of inversions in the list.
    """
    inversions = 0
    for l in range(1, len(xs)):
        for i in range(l, 0, -1):
            if xs[i] < xs[i-1]:
                xs[i-1], xs[i] = xs[i], xs[i-1]
                inversions += 1
            else:
                break
    return inversions
