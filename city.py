def efficient_city(N, influences):
    """
    This function checks if the city is highly efficient by verifying that every pair of citizens can trade within the overlapping 
    spheres of influence. It either returns "Yes" if the city is efficient, or "No i j" indicating the first pair of citizens 
    (i, j) that cannot perform an efficient trade.
    
    How the algorithm works:
    1. **Initialization**: Each citizen has a sphere of influence [L_i, R_i]. The function checks if these spheres overlap
       for every citizen i and their neighbor i+1.
    2. **Efficient trade check**: For the city to be highly efficient, every citizen's sphere of influence must be continuous
       across all citizens. This means that the right endpoint of each citizen must overlap or extend with the left endpoint 
       of their neighbors.
    3. **Output**: The function returns "Yes" if all pairs can trade efficiently, or "No" followed by the first pair of citizens
       that cannot trade efficiently.

    Time Complexity: O(N), where N is the number of citizens (or houses), as the algorithm processes each citizen once.
    Space Complexity: O(1) extra space aside from the input.

    Parameters:
    - N: Number of citizens.
    - influences: List of tuples representing each citizen's sphere of influence [L, R].

    Returns:
    - "Yes" if the city is highly efficient, otherwise "No i j" where i and j are the first pair of citizens who cannot trade.
    """

    # Initialize minimum left boundary and maximum right boundary
    left, right = influences[0]  # Start with the first citizen's sphere of influence

    for i in range(1, N):
        L, R = influences[i]  # Current citizen's sphere of influence
        
        # Check if there's an overlap between current and previous spheres of influence
        if L > right:
            # No overlap found, return the first pair that cannot trade efficiently
            return f"No {i} {i + 1}"
        
        # Update the boundaries to maintain the largest continuous region
        left = max(left, L)
        right = min(right, R)

    # If all pairs can trade efficiently
    return "Yes"

# Input
N = int(input())
influences = [tuple(map(int, input().split())) for _ in range(N)]

# Output the result
print(efficient_city(N, influences))
