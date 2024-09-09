def ceil_sqrt(x):
    """Calculate the ceiling of the square root of x manually."""
    root = int(x ** 0.5)  # Get the integer part of the square root
    if root * root == x:
        return root
    return root + 1

def eating_bacteria(N, S, H, bacteria_sizes):
    """
    This function calculates the maximum size you can reach after consuming bacteria for a given number of hours.

    Problem Breakdown:
    - You can consume a bacterium if its size is strictly smaller than yours.
    - When you consume a bacterium of size `s2`, your size increases by the ceiling of the square root of `s2`.
    - You have a maximum number of H hours to consume bacteria, and each hour you can consume one bacterium.

    Approach:
    1. Sort the bacteria sizes in ascending order.
    2. For each hour (up to H), consume the smallest bacterium that you can (one that is smaller than your size).
    3. Update your size by the ceiling of the square root of the consumed bacterium.
    4. Stop if no more consumable bacteria are left.

    Time Complexity:
    - O(N log N) for sorting and processing the bacteria.

    Space Complexity:
    - O(N) for storing the list of bacteria.

    Parameters:
    - N: The number of bacteria in the Petri dish.
    - S: Your initial size.
    - H: The number of hours you have to consume bacteria.
    - bacteria_sizes: A list of integers representing the sizes of the other bacteria.

    Returns:
    - The maximum size you can achieve after consuming bacteria for H hours.
    """
    
    # Sort bacteria sizes in ascending order
    bacteria_sizes.sort()
    
    # Consume bacteria for H hours or until no smaller bacteria are left
    consumed = 0
    while consumed < H:
        # Find the smallest bacterium that you can consume
        index = -1
        for i in range(len(bacteria_sizes)):
            if bacteria_sizes[i] < S:
                index = i
                break

        if index == -1:
            break  # No more bacteria to consume

        # Consume the smallest bacterium and update size
        smallest = bacteria_sizes.pop(index)
        S += ceil_sqrt(smallest)
        consumed += 1

    return S

# Input
N, S, H = map(int, input().split())
bacteria_sizes = list(map(int, input().split()))

# Output the maximum size you can reach
print(eating_bacteria(N, S, H, bacteria_sizes))
