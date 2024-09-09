"""
List of Problems and Their Time and Space Complexities (with explanations and use cases):

1. **Kameleon Maze** (BFS):
   - **Explanation**: Uses BFS to find the shortest path to eat candies in increasing order in a maze.
   - **Use Case**: Use when searching for the shortest path in a grid-like structure.

   - **Time Complexity**: O(r * c), where r is the number of rows and c is the number of columns in the maze.
   - **Space Complexity**: O(r * c), for the queue and visited structures.

2. **Go Stone Puzzle** (DFS):
   - **Explanation**: Uses DFS to check if we can achieve a target stone configuration by moving pairs of stones.
   - **Use Case**: Use when solving puzzles involving depth-first exploration.

   - **Time Complexity**: O(N!), where N is the number of stones (items in the puzzle).
   - **Space Complexity**: O(N), due to recursion depth and storage of visited states.

3. **Bare Bones BFS**:
   - **Explanation**: Basic BFS template to traverse all nodes of a graph.
   - **Use Case**: Use when exploring all neighbors of a node in a graph.

   - **Time Complexity**: O(V + E), where V is the number of nodes and E is the number of edges.
   - **Space Complexity**: O(V), for the queue and visited set.

4. **Bare Bones DFS**:
   - **Explanation**: Basic DFS template to explore all paths in a graph.
   - **Use Case**: Use when exploring all paths or solving connectivity problems.

   - **Time Complexity**: O(V + E), where V is the number of nodes and E is the number of edges.
   - **Space Complexity**: O(V), for the recursion stack and visited set.

5. **Mountain Waterfall** (DFS):
   - **Explanation**: Simulates water flowing down a mountain using DFS.
   - **Use Case**: Use for flow or connectivity problems in a grid structure.

   - **Time Complexity**: O(R * C), where R is the number of rows and C is the number of columns in the grid.
   - **Space Complexity**: O(R * C), for the grid and DFS stack.

6. **Efficient City** (Graph Traversal - BFS):
   - **Explanation**: Checks if the city is efficient by ensuring every citizen can trade with all others via BFS.
   - **Use Case**: Use when checking reachability and connectivity in a graph.

   - **Time Complexity**: O(N), where N is the number of citizens (platforms).
   - **Space Complexity**: O(N), for the in-degree array and queue.

7. **Flavours Galore** (Topological Sort - DP):
   - **Explanation**: Uses topological sort to find the longest chain of acceptable flavors.
   - **Use Case**: Use when solving dependency-based problems in DAGs.

   - **Time Complexity**: O(F + P), where F is the number of flavors and P is the number of pairs.
   - **Space Complexity**: O(F + P), for storing the graph and DP arrays.

8. **GPU Meme** (Knapsack-like Dynamic Programming):
   - **Explanation**: Solves a problem similar to knapsack where we maximize environmental cost within legal limits.
   - **Use Case**: Use when optimizing for multiple constraints with DP.

   - **Time Complexity**: O(N * E), where N is the number of GPUs and E is the maximum environmental cost.
   - **Space Complexity**: O(E), for the DP array.

9. **Master of Jenga** (Greedy):
   - **Explanation**: Places blocks in a grid to maximize score using a greedy approach.
   - **Use Case**: Use when maximizing values in grid problems.

   - **Time Complexity**: O(R * (C - k + 1)), where R is the number of rows, C is the number of columns, and k is the width of the brick.
   - **Space Complexity**: O(R * C), for the result grid.

10. **Crushing Monsters** (Knapsack-like DP):
    - **Explanation**: Solves the problem of minimizing the boss's HP using potions and swords, similar to knapsack.
    - **Use Case**: Use for minimizing or maximizing in resource-limited problems.

    - **Time Complexity**: O(N log N), where N is the number of items (potions and swords). Sorting takes O(N log N).
    - **Space Complexity**: O(N), for storing the items.

11. **Distinct Platforms** (Sorting and Greedy):
    - **Explanation**: Determines if all platforms can be visited exactly once by jumping between overlapping ranges.
    - **Use Case**: Use when solving problems involving intervals and overlaps.

    - **Time Complexity**: O(N log N), where N is the number of platforms. Sorting the platforms dominates the time.
    - **Space Complexity**: O(N), for storing the platforms and their ranges.

12. **Eating Bacteria** (Greedy):
    - **Explanation**: Maximizes your size by consuming bacteria smaller than you for a limited number of hours.
    - **Use Case**: Use for greedy problems where you can grow by consuming resources.

    - **Time Complexity**: O(N log N), where N is the number of bacteria. Sorting bacteria takes O(N log N).
    - **Space Complexity**: O(N), for storing bacteria sizes.

13. **Maximum Sum Subarray (Kadane's Algorithm)**:
    - **Explanation**: Finds the maximum sum of a contiguous subarray using Kadane's algorithm.
    - **Use Case**: Use for maximizing or minimizing sum-based subarray problems.

    - **Time Complexity**: O(N), where N is the number of integers in the array. One pass through the array.
    - **Space Complexity**: O(1), as we only use a few variables.

14. **Maximum Sum Subarray (Brute Force)**:
    - **Explanation**: Finds the maximum sum of a contiguous subarray using brute force by calculating all subarrays.
    - **Use Case**: Use when trying to understand subarray sum calculations.

    - **Time Complexity**: O(N^2), where N is the number of integers in the array. We calculate the sum of every subarray.
    - **Space Complexity**: O(1), as we only use a few variables.

15. **0-1 Knapsack Problem** (Dynamic Programming):
    - **Explanation**: Solves the 0-1 knapsack problem where we choose items to maximize value without exceeding weight.
    - **Use Case**: Use when solving resource optimization problems with weight constraints.

    - **Time Complexity**: O(N * W), where N is the number of items and W is the maximum weight.
    - **Space Complexity**: O(N * W), for storing the DP table.

16. **Fractional Knapsack** (Greedy Algorithm):
    - **Explanation**: Solves the fractional knapsack problem by taking as much of the highest value-to-weight ratio items.
    - **Use Case**: Use when solving problems where fractions of items can be taken.

    - **Time Complexity**: O(N log N), where N is the number of items. Sorting items takes O(N log N).
    - **Space Complexity**: O(N), for storing the items.

---

Additional Graph Theory and Classic Algorithms:

17. **Hamiltonian Path (Backtracking)**:
    - **Explanation**: Finds a path that visits every vertex exactly once using backtracking.
    - **Use Case**: Use when solving path traversal problems in graphs.

    - **Time Complexity**: O(N!), where N is the number of vertices.
    - **Space Complexity**: O(N), for storing the path and visited nodes.

18. **Dijkstra's Algorithm** (Single Source Shortest Path in Graphs):
    - **Explanation**: Finds the shortest path from a source node to all other nodes in a weighted graph.
    - **Use Case**: Use when solving shortest path problems in weighted graphs with non-negative edges.

    - **Time Complexity**: O((V + E) log V), where V is the number of vertices and E is the number of edges.
    - **Space Complexity**: O(V + E), for the adjacency list and distance table.

19. **Prim's Algorithm** (Minimum Spanning Tree):
    - **Explanation**: Finds the minimum spanning tree of a graph using a greedy approach.
    - **Use Case**: Use when finding the minimum cost to connect all nodes in a graph.

    - **Time Complexity**: O((V + E) log V), where V is the number of vertices and E is the number of edges.
    - **Space Complexity**: O(V + E), for storing the graph and minimum spanning tree information.

20. **Bellman-Ford Algorithm** (Single Source Shortest Path for Graphs with Negative Weights):
    - **Explanation**: Finds the shortest path from a source node to all other nodes in a graph with negative edges.
    - **Use Case**: Use when solving shortest path problems in graphs with negative weights.

    - **Time Complexity**: O(V * E), where V is the number of vertices and E is the number of edges.
    - **Space Complexity**: O(V), for storing distances.

21. **Floyd-Warshall Algorithm** (All-Pairs Shortest Path):
    - **Explanation**: Solves the all-pairs shortest path problem for graphs.
    - **Use Case**: Use when solving shortest paths between all pairs of nodes in a graph.

    - **Time Complexity**: O(V^3), where V is the number of vertices.
    - **Space Complexity**: O(V^2), for storing the distance matrix.

22. **Merge Sort** (Sorting Algorithm):
    - **Explanation**: Sorts an array using divide-and-conquer by merging two sorted halves.
    - **Use Case**: Use when stable sorting or sorting large data efficiently.

    - **Time Complexity**: O(N log N), where N is the number of elements.
    - **Space Complexity**: O(N), due to the auxiliary space required for merging.

23. **Quick Sort** (Sorting Algorithm):
    - **Explanation**: Sorts an array using divide-and-conquer by partitioning around a pivot.
    - **Use Case**: Use when in-place sorting is required, though it can degrade in performance.

    - **Time Complexity**: O(N log N) on average, O(N^2) in the worst case.
    - **Space Complexity**: O(log N) due to recursion stack, O(N) in the worst case for unbalanced partitions.

24. **Binary Search** (Searching Algorithm):
    - **Explanation**: Finds an element in a sorted array by dividing the search space in half.
    - **Use Case**: Use when searching in a sorted array.

    - **Time Complexity**: O(log N), where N is the number of elements.
    - **Space Complexity**: O(1), as no additional memory is needed aside from a few variables.

25. **Topological Sort** (Ordering of Directed Acyclic Graph):
    - **Explanation**: Finds a linear ordering of vertices in a directed acyclic graph.
    - **Use Case**: Use when solving dependency ordering problems in DAGs.

    - **Time Complexity**: O(V + E), where V is the number of vertices and E is the number of edges.
    - **Space Complexity**: O(V), for storing the sorted order and adjacency list.

26. **Union-Find (Disjoint Set Union)**:
    - **Explanation**: Efficiently tracks connected components of elements using union-by-rank and path compression.
    - **Use Case**: Use when solving connectivity or component problems in graphs.

    - **Time Complexity**: O(α(N)), where α is the inverse Ackermann function, which is very small, nearly constant.
    - **Space Complexity**: O(N), for storing parent and rank arrays.

27. **Tarjan's Algorithm** (Finding Strongly Connected Components):
    - **Explanation**: Finds all strongly connected components in a directed graph using DFS.
    - **Use Case**: Use when solving connectivity or component problems in directed graphs.

    - **Time Complexity**: O(V + E), where V is the number of vertices and E is the number of edges.
    - **Space Complexity**: O(V), for storing discovery times, low values, and the recursion stack.

28. **Kruskal's Algorithm** (Minimum Spanning Tree using Union-Find):
    - **Explanation**: Finds the minimum spanning tree by sorting edges and using union-find.
    - **Use Case**: Use when constructing a minimum cost tree for connecting nodes.

    - **Time Complexity**: O(E log E), where E is the number of edges (sorting the edges).
    - **Space Complexity**: O(V + E), for storing edges, parent array, and rank array.

29. **Ford-Fulkerson Algorithm** (Maximum Flow in a Network):
    - **Explanation**: Finds the maximum flow in a network by augmenting paths iteratively.
    - **Use Case**: Use when solving flow network problems like scheduling and resource allocation.

    - **Time Complexity**: O(E * f), where E is the number of edges and f is the maximum flow.
    - **Space Complexity**: O(V + E), for storing capacities and residual graphs.

30. **Edmonds-Karp Algorithm** (Implementation of Ford-Fulkerson using BFS):
    - **Explanation**: Finds the maximum flow in a network using BFS to find augmenting paths.
    - **Use Case**: Use for finding max flow in networks where BFS performs better.

    - **Time Complexity**: O(V * E^2), where V is the number of vertices and E is the number of edges.
    - **Space Complexity**: O(V + E), for storing the graph and flow values.

"""
