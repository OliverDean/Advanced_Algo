def crushing_monsters(h, n, m, L, potions, swords):
    """
    This function calculates the minimum HP the boss can have after Jimothy uses a limited number of items (potions and swords)
    in any order.

    Problem Overview:
    - Jimothy can use up to L items from his collection of potions and swords to reduce the boss's HP.
    - Potions reduce the boss's HP by a percentage of its current HP, while swords reduce the boss's HP by a fixed amount.
    - The goal is to minimize the boss's HP after using up to L items.

    Approach:
    1. Sort the potions in decreasing order of potency (the higher the potency, the better).
    2. Sort the swords in decreasing order of strength (the higher the sword strength, the better).
    3. Consider all combinations of using up to L potions and swords, and simulate their effect on the boss's HP.
    4. Track the minimum HP achieved in all valid combinations.

    Time Complexity:
    - Sorting the potions and swords is O(n log n + m log m), where n is the number of potions and m is the number of swords.
    - Simulating the effects for all combinations is O(L), so the overall complexity is efficient for the input constraints.

    Parameters:
    - h: Initial HP of the boss.
    - n: Number of potions.
    - m: Number of swords.
    - L: Maximum number of items Jimothy can use.
    - potions: List of potion potencies.
    - swords: List of sword strengths.

    Returns:
    - The minimum HP the boss can have after using up to L items.
    """

    # Sort potions and swords in decreasing order
    potions.sort(reverse=True)
    swords.sort(reverse=True)
    
    min_hp = float('inf')  # Track the minimum HP achievable

    # Consider using up to min(n, L) potions and the rest as swords
    for potion_count in range(min(n, L) + 1):
        current_hp = h
        # Apply potions
        for i in range(potion_count):
            current_hp *= (1 - potions[i] / 100.0)
        # Apply swords for the remaining items
        sword_count = L - potion_count
        for j in range(min(sword_count, m)):
            current_hp -= swords[j]
        # Update minimum HP found
        min_hp = min(min_hp, current_hp)

    return min_hp

# Input reading
h, n, m, L = map(int, input().split())
potions = list(map(int, input().split()))
swords = list(map(int, input().split()))

# Output the minimum HP the boss can have
print(f"{crushing_monsters(h, n, m, L, potions, swords):.10f}")
