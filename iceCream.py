


"""
    This function determines the maximum number of scoops in the tallest ice cream Kiki can make based on flavour 
    compatibility or if it's possible to build an arbitrarily tall ice cream tower due to cycles in the flavour hierarchy.

    Problem Overview:
    - Kiki can stack different flavours of ice cream, but certain flavours are only acceptable directly on top of specific others.
    - The input describes the acceptable flavour pairs as directed relationships between flavours.
    - If the relationships form a cycle, Kiki can build an arbitrarily tall ice cream tower.
    - Otherwise, the problem reduces to finding the longest path in a Directed Acyclic Graph (DAG).

    Algorithm Explanation:
    1. **Graph Representation**:
       - Flavours are represented as nodes, and each pair (x, y) indicates a directed edge from node y to node x (i.e., flavour x can go on top of flavour y).
       - An `in_degree` array tracks how many incoming edges (dependencies) each flavour has.
    
    2. **Topological Sorting (Kahn's Algorithm)**:
       - To find the longest path and detect cycles, we perform a topological sort using Kahn's algorithm.
       - We start with all nodes that have an in-degree of 0 (flavours with no dependencies) and process them by reducing the in-degree of their neighbours.
       - As we process the graph, we track the longest path from any starting node (flavour) using a `longest_path` array.
    
    3. **Cycle Detection**:
       - If the topological sorting fails to include all nodes, it means there is a cycle in the graph (i.e., a flavour depends on itself indirectly), so an arbitrarily tall tower is possible.
       - In such a case, we return `-1`.

    4. **Longest Path Calculation**:
       - If no cycle exists, the longest path in the graph is calculated by updating the `longest_path` array during the topological sort. The maximum value in this array represents the tallest possible ice cream tower.
    
    Parameters:
    - F: The number of flavours (nodes).
    - P: The number of acceptable pairs of flavours (edges).
    - pairs: A list of tuples representing acceptable pairs (x, y), meaning flavour x can be placed on top of flavour y.

    Returns:
    - An integer representing the maximum number of scoops in the tallest ice cream tower or `-1` if the tower can be arbitrarily tall (due to cycles).

    Time Complexity:
    - O(F + P), where F is the number of flavours (nodes) and P is the number of pairs (edges), since we process each node and edge once in the topological sort.

    Space Complexity:
    - O(F + P) for storing the graph and other structures like the in-degree array and longest path array.

    Example:
    Input:
    3
    2
    1 2
    2 3

    Output:
    3
    Explanation: Kiki can build a tower of flavours 3 -> 2 -> 1, resulting in 3 scoops.

    Input:
    3
    3
    1 2
    2 3
    3 1

    Output:
    -1
    Explanation: There is a cycle, so Kiki can build an arbitrarily tall tower.
"""



from collections import defaultdict, deque

def flavours_galore(F, P, pairs):
    # Build the graph and in-degrees
    graph = defaultdict(list)
    in_degree = [0] * (F + 1)

    for x, y in pairs:
        graph[y].append(x)  # y -> x (x can be on top of y)
        in_degree[x] += 1   # Increment in-degree of x
    
    # Perform topological sort (Kahn's algorithm)
    queue = deque()
    for i in range(1, F + 1):
        if in_degree[i] == 0:
            queue.append(i)  # Add all nodes with in-degree 0 to the queue
    
    topo_order = []
    longest_path = [0] * (F + 1)
    
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        
        # For each neighbour, update the longest path and decrease in-degree
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            longest_path[neighbor] = max(longest_path[neighbor], longest_path[node] + 1)
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check if there was a cycle (i.e., not all nodes were processed)
    if len(topo_order) < F:
        return -1  # There is a cycle, so the number of scoops can be arbitrarily large
    
    # Return the length of the longest path + 1 (since each path is a sequence of scoops)
    return max(longest_path)

# Input
F = int(input())  # Number of flavours
P = int(input())  # Number of acceptable pairs
pairs = [tuple(map(int, input().split())) for _ in range(P)]

# Output
print(flavours_galore(F, P, pairs))
