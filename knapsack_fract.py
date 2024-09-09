def fractional_knapsack(N, W, items):
    """
    This function solves the Fractional Knapsack problem using a greedy approach.

    Problem Breakdown:
    - You are given N items, each with a weight and a value.
    - The goal is to maximize the total value that can be carried in a knapsack with a maximum weight limit W.
    - Unlike the 0-1 knapsack problem, you can take fractions of items in this problem, meaning you can take 
      any portion of an item to maximize value.

    Approach:
    1. Sort the items by their value-to-weight ratio in descending order.
    2. Greedily take as much as possible of the item with the highest ratio. If the knapsack has remaining capacity, 
       continue with the next item.
    3. For each item, if the remaining capacity is less than the item's weight, take only a fraction of the item.
    4. Stop when the knapsack is full or when all items are processed.

    Time Complexity: O(N log N), where N is the number of items. This is due to the sorting step.
    Space Complexity: O(N), for storing the items.

    Parameters:
    - N: The number of items.
    - W: The maximum weight capacity of the knapsack.
    - items: A list of tuples (value, weight) representing the value and weight of each item.

    Returns:
    - The maximum total value that can be carried in the knapsack.
    """

    # Sort items by value-to-weight ratio in descending order
    items.sort(key=lambda x: x[0] / x[1], reverse=True)

    total_value = 0  # To store the total value we can carry
    remaining_capacity = W

    for value, weight in items:
        if remaining_capacity >= weight:
            # If we can take the whole item, take it
            total_value += value
            remaining_capacity -= weight
        else:
            # If we can take only part of the item, take the fraction of it
            total_value += value * (remaining_capacity / weight)
            break  # The knapsack is full

    return total_value

# Input
N, W = map(int, input().split())
items = [tuple(map(int, input().split())) for _ in range(N)]

# Output the maximum value we can carry
print(f"{fractional_knapsack(N, W, items):.6f}")
