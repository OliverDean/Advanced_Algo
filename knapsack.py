def knapsack(N, W, weights, values):
    """
    This function solves the 0-1 Knapsack Problem using dynamic programming.

    Problem Breakdown:
    - We are given N items, each with a weight and a value.
    - The goal is to maximize the value we can carry in a knapsack with a maximum weight capacity W.
    - For each item, we can either include it in the knapsack or exclude it (hence 0-1 knapsack).

    Approach:
    1. Use dynamic programming with a 2D dp array:
       - dp[i][w] represents the maximum value achievable with the first i items and a weight limit of w.
    2. For each item, either:
       - Exclude it: the value remains the same as dp[i-1][w].
       - Include it: add its value and check if we can fit it based on its weight.
    3. Use the recurrence relation:
       - dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i]] + values[i]) if the item fits, else dp[i][w] = dp[i-1][w].

    Time Complexity: O(N * W), where N is the number of items and W is the maximum weight capacity.
    Space Complexity: O(N * W), for the dp array.

    Parameters:
    - N: The number of items.
    - W: The maximum weight capacity of the knapsack.
    - weights: A list of integers representing the weights of the items.
    - values: A list of integers representing the values of the items.

    Returns:
    - The maximum value that can be carried in the knapsack.
    """

    # Initialize dp array with 0s. dp[i][w] represents max value with first i items and weight limit w
    dp = [[0] * (W + 1) for _ in range(N + 1)]

    # Fill the dp table
    for i in range(1, N + 1):
        for w in range(W + 1):
            if weights[i - 1] <= w:  # Check if the item can be included
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[N][W]  # The bottom-right value is the maximum value achievable

# Input
N, W = map(int, input().split())
weights = list(map(int, input().split()))
values = list(map(int, input().split()))

# Output the maximum value that can be carried in the knapsack
print(knapsack(N, W, weights, values))
