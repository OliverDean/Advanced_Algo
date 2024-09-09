from collections import deque


"""
    The solution uses Breadth-First Search (BFS) to compute the shortest path from Pascal's start position to each candy in increasing order.
    For each candy, the BFS explores all reachable cells within Pascal's tongue range (k) in the four cardinal directions,
    ensuring blocked cells are avoided. The process repeats for all nine candies, accumulating the time taken.
    The time complexity is O(r⋅c) for each BFS call, leading to a total time complexity of O(9⋅r⋅c), which simplifies to O(r⋅c).
    The space complexity is also O(r⋅c)due to the grid and visited structures used in BFS.
"""

# Input
r, c, k = map(int, input().split())
grid = [input().strip() for _ in range(r)]

# Directions (up, down, left, right)
deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Find Pascal's start and candy locations
start = None
candies = [None] * 9
for i in range(r):
    for j in range(c):
        if grid[i][j] == 'S':
            start = (i, j)
        elif grid[i][j].isdigit():
            candies[int(grid[i][j]) - 1] = (i, j)

# BFS to find the shortest time to move and eat candies
def bfs(start, end):
    queue = deque([(start[0], start[1], 0)])  # (row, col, distance)
    visited = [[False] * c for _ in range(r)]
    visited[start[0]][start[1]] = True
    
    while queue:
        x, y, dist = queue.popleft()
        if (x, y) == end:
            return dist
        
        for dx, dy in deltas:
            for step in range(1, k+1):
                nx, ny = x + dx * step, y + dy * step
                if 0 <= nx < r and 0 <= ny < c and grid[nx][ny] != '#' and not visited[nx][ny]:
                    visited[nx][ny] = True
                    queue.append((nx, ny, dist + 1))
                if grid[nx][ny] != '.':
                    break

    return float('inf')

# Compute the total time to collect candies in order
total_time = 0
current_position = start
for candy in candies:
    total_time += bfs(current_position, candy)
    current_position = candy

# Output the result
print(total_time)
