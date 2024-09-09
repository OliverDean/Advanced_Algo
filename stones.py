
"""
    Input Parsing: The number of stones N, initial state S, and target state T are read.
Feasibility Check: Before attempting any moves, we check if sorting S and T results in the same sequence of stones. If they don't match, output -1 as its impossible to reach the target.
BFS Approach: Using BFS, the algorithm attempts all possible moves, maintaining a queue of states (current configuration of the stones) and the number of moves taken to reach that state. For each state, it explores moving adjacent pairs of stones to the two empty cells, and the process continues until the target state is reached.
Visited Set: A set is used to track visited states to avoid repeating the same configurations.
Time Complexity: The BFS explores all possible states, and the number of possible states is small due to N ≤ 14. Hence, the time complexity is approximately O((N+2)!), but with BFS, the solution is efficient for small N.
"""


from collections import deque

# Helper function to count operations needed
def min_operations(N, start, target):
    start = list(start)
    target = list(target)

    # Check if transformation is even possible by comparing stone counts
    if sorted(start) != sorted(target):
        return -1

    # BFS to find the minimum number of operations
    queue = deque([(start, 0)])
    visited = set()
    visited.add(tuple(start))
    
    while queue:
        current, moves = queue.popleft()
        
        # If current matches target, return number of moves
        if current[:N] == target:
            return moves
        
        # Try all valid adjacent pairs to move
        for i in range(N - 1):
            if current[i] != '.' and current[i + 1] != '.':
                # Find the empty spots
                empty1, empty2 = N, N + 1
                
                # Swap the stones from i and i + 1 with empty1 and empty2
                new_state = current[:]
                new_state[empty1], new_state[empty2] = new_state[i], new_state[i + 1]
                new_state[i], new_state[i + 1] = '.', '.'

                # Convert to tuple to store in visited set
                new_state_tuple = tuple(new_state)
                
                if new_state_tuple not in visited:
                    visited.add(new_state_tuple)
                    queue.append((new_state, moves + 1))

    return -1

# Input
N = int(input())
S = input().strip()
T = input().strip()

# Initial setup with empty cells at the end
S = S + ".."
T = T + ".."

# Output the result
print(min_operations(N, S, T))
