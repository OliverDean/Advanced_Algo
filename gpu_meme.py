def gpu_meme(N, E, GPUs):
    """
    This function solves the problem using a dynamic programming (DP) approach similar to the knapsack problem.
    
    Problem Overview:
    - Varun wants to buy a subset of GPUs such that the total environmental cost of the GPUs does not exceed a legal limit E.
    - He aims to maximize the total environmental cost while minimizing the monetary cost if there are multiple ways to reach the same environmental cost.

    Approach:
    - We use dynamic programming to keep track of the minimum monetary cost needed to achieve a certain environmental cost.
    - Let dp[i] represent the minimum monetary cost to achieve an environmental cost of exactly i.
    - We initialize dp[0] = 0 (no monetary cost to achieve 0 environmental cost) and all other dp[i] to infinity (because it's initially impossible to achieve those environmental costs).
    - For each GPU, we update the dp array by considering if including the GPU leads to a better solution (in terms of minimizing monetary cost for the same environmental cost).
    
    Steps:
    1. We initialize the dp array with a size of E+1 (the legal limit plus 1) and fill it with infinity (except for dp[0] = 0).
    2. For each GPU, we update the dp array from the back (starting from the environmental limit down to the GPU's environmental cost). This ensures we do not double count a GPU.
    3. After processing all GPUs, we find the maximum environmental cost we can achieve within the limit E.
    4. If multiple subsets achieve the same environmental cost, we choose the one with the minimal monetary cost.
    
    Time Complexity:
    - O(N * E), where N is the number of GPUs and E is the environmental cost limit. We process each GPU and update the dp array in O(E) for each GPU.

    Space Complexity:
    - O(E) due to the dp array that tracks the minimum monetary cost for each environmental cost.
    
    Parameters:
    - N: Number of GPUs.
    - E: Legal limit on the total environmental cost.
    - GPUs: List of tuples, where each tuple contains (M, C) representing the monetary cost and environmental cost of each GPU.

    Returns:
    - The minimum monetary cost to achieve the maximum possible environmental cost within the limit.
    """
    
    # Initialize dp array to store the minimum monetary cost for each environmental cost
    dp = [float('inf')] * (E + 1)
    dp[0] = 0  # It takes 0 money to have an environmental cost of 0
    
    # Process each GPU
    for M, C in GPUs:
        for env_cost in range(E, C - 1, -1):  # Update dp from the back to avoid double counting
            dp[env_cost] = min(dp[env_cost], dp[env_cost - C] + M)
    
    # Find the maximum environmental cost that can be achieved within the limit E
    max_env_cost = max(i for i in range(E + 1) if dp[i] < float('inf'))
    
    # Return the minimum monetary cost to achieve that maximum environmental cost
    return dp[max_env_cost]

# Input
N, E = map(int, input().split())
GPUs = [tuple(map(int, input().split())) for _ in range(N)]

# Output
print(gpu_meme(N, E, GPUs))
