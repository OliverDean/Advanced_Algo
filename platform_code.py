
"""
    This function determines if it's possible to visit all platforms exactly once by jumping between platforms 
    where their ranges overlap. If possible, it returns the order of platforms to visit, otherwise, it returns "NO".
    
    Problem Breakdown:
    - Each platform has a range [L_i, R_i].
    - You can jump between platforms if their ranges overlap, i.e., there exists some integer x such that:
      L_a <= x <= R_a and L_b <= x <= R_b, where platforms a and b overlap.
    - We need to determine if there is an ordering of platforms where we can visit all platforms exactly once.

    Approach:
    1. **Sorting by Range**:
       - Sort platforms by their left endpoint (L_i) to attempt visiting them in order of their start points.
       - This approach helps because if we can keep jumping from the platform with the lowest left endpoint to the next, 
         we increase our chances of covering all platforms without getting stuck.
    2. **Greedy Checking**:
       - After sorting, we attempt to traverse through the sorted platforms. If at any point the current platform’s range 
         does not overlap with the next platform's range, it means we cannot jump between them, and the answer is "NO".
    3. **Time Complexity**:
       - Sorting the platforms takes O(N log N), which is efficient enough for N ≤ 500,000.

    Time Complexity: O(N log N) due to sorting.
    Space Complexity: O(N) for storing the platform list and the result.

    Parameters:
    - N: The number of platforms.
    - platforms: A list of tuples representing the platforms, where each tuple is (L, R).

    Returns:
    - A string "YES" followed by a valid order of platforms, or "NO" if it's not possible to visit all platforms exactly once.
    """
def distinct_platforms(N, platforms):
    # Add index to track the original position of platforms
    platforms_with_index = sorted((L, R, i + 1) for i, (L, R) in enumerate(platforms))

    # Check if the platforms can be visited in a valid order
    for i in range(N - 1):
        _, R1, _ = platforms_with_index[i]
        L2, _, _ = platforms_with_index[i + 1]
        if R1 < L2:  # No overlap between the current platform and the next one
            return "NO"

    # If we can visit all platforms in a valid order, return the order
    return "YES\n" + " ".join(str(platform[2]) for platform in platforms_with_index)

# Input reading
N = int(input())
platforms = [tuple(map(int, input().split())) for _ in range(N)]

# Output the result
print(distinct_platforms(N, platforms))
