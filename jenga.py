def master_of_jenga(R, C, k, grid):
    """
    This function determines a valid brick configuration to maximize the points Jessica can score in the 2D-Jenga game.
    
    Problem Overview:
    - The game is played on an R x C grid where each brick has a width of k and height of 1.
    - Each row must have one brick, and the brick must balance on the row directly below it.
    - A brick balances if at least one column of the brick overlaps with a brick directly below it.
    - The goal is to maximize the sum of the values covered by the bricks in the grid.

    Approach:
    1. For each row, choose a brick of width k that maximizes the sum of the values covered by that brick in that row.
    2. For simplicity, the code selects a brick that starts at the leftmost possible position of the highest sum.
    3. Since balancing bricks only requires overlapping one column, a greedy approach that maximizes each row works.
    
    Steps:
    1. For each row, slide the brick of length k across the row and calculate the sum of values covered by the brick.
    2. Track the position of the brick that gives the highest score for each row.
    3. Output the grid with the brick configuration.

    Time Complexity:
    - O(R * (C - k + 1)), where R is the number of rows, and C is the number of columns. We calculate the sum for each possible position of a brick in each row.

    Space Complexity:
    - O(R * C), the space used for storing the grid and the result.

    Parameters:
    - R: The number of rows in the grid.
    - C: The number of columns in the grid.
    - k: The width of each brick.
    - grid: A 2D list representing the values in each cell of the grid.

    Returns:
    - A list of strings representing the grid with bricks placed ('X' for brick and '.' for empty space).
    """

    result = [['.'] * C for _ in range(R)]  # Initialize the result grid with empty cells

    for r in range(R):
        # Find the best position for the brick in row r
        best_start = 0
        best_sum = sum(grid[r][0:k])  # Initial sum of the first k elements

        # Slide the brick across the row to find the optimal position
        for c in range(1, C - k + 1):
            current_sum = sum(grid[r][c:c + k])
            if current_sum > best_sum:
                best_sum = current_sum
                best_start = c

        # Place the brick at the best position
        for c in range(best_start, best_start + k):
            result[r][c] = 'X'

    # Output the result as strings
    for row in result:
        print(''.join(row))

# Input
R, C, k = map(int, input().split())
grid = [list(map(int, input().split())) for _ in range(R)]

# Output the valid configuration of bricks
master_of_jenga(R, C, k, grid)
