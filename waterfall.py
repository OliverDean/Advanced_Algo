def simulate_waterfall(grid, R, C):
    """
    This function simulates the flow of water down a hill based on the rules of the problem. Water flows vertically until it
    hits a rock, at which point it splits left and right. It keeps flowing left and right, and down whenever it encounters an
    empty cell ('.') below it.

    How DFS-based Waterfall Simulation Works:
    1. **Water Flow**: Starting from each water source ('*'), water flows down vertically until a rock ('o') is encountered.
    2. **Splitting Flow**: When water hits a rock directly below it, it flows left and right until it either encounters another
       rock or finds an empty cell ('.') below to flow down further.
    3. **Recursive DFS**: DFS is used to simulate this process by recursively exploring downward, leftward, and rightward
       flows when water encounters a rock below.
    4. **Boundary Handling**: The boundary of the grid is always filled with rocks, ensuring water cannot flow out of bounds.

    Characteristics:
    - The time complexity is O(R * C), where R is the number of rows and C is the number of columns.
    - The space complexity is O(R * C), primarily due to the recursion stack and the updated grid.

    Parameters:
    - grid: 2D list of characters representing the hill (rocks, water, empty spaces)
    - R: Number of rows in the grid
    - C: Number of columns in the grid
    
    Returns:
    - The updated grid after simulating the water flow.
    """
    def flow(x, y):
        # Flow down until a rock or boundary is hit
        while x + 1 < R and grid[x + 1][y] == '.':
            grid[x + 1][y] = '*'
            x += 1
        
        # Now flow left and right if hit a rock below
        if x + 1 < R and grid[x + 1][y] == 'o':
            # Flow left
            l = y - 1
            while l >= 0 and grid[x][l] == '.' and grid[x + 1][l] != '.':
                grid[x][l] = '*'
                l -= 1
            # Flow right
            r = y + 1
            while r < C and grid[x][r] == '.' and grid[x + 1][r] != '.':
                grid[x][r] = '*'
                r += 1

    # Iterate over the grid and trigger water flow for every water source ('*')
    for i in range(R):
        for j in range(C):
            if grid[i][j] == '*':
                flow(i, j)

    return grid

# Input
R, C = map(int, input().split())
grid = [list(input().strip()) for _ in range(R)]

# Simulate waterfall
result_grid = simulate_waterfall(grid, R, C)

# Output the result
for row in result_grid:
    print(''.join(row))
