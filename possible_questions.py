
"""
List of Algorithm Names in Order:

1. **Two Sum (HashMap Approach)**:
   - Explanation: Given an array of integers and a target sum, find two numbers in the array that add up to the target.

2. **Palindrome Check (Two Pointer Approach)**:
   - Explanation: Check if a given string is a palindrome by comparing characters from both ends.

3. **Fibonacci Sequence (Dynamic Programming)**:
   - Explanation: Compute the N-th Fibonacci number using DP to avoid recomputation.

4. **Find the Missing Number (XOR Approach)**:
   - Explanation: Given an array of numbers from 1 to N, find the missing number.

5. **Binary Search (Iterative)**:
   - Explanation: Perform a binary search on a sorted array to find a target value.

6. **Maximum Subarray Sum (Kadane's Algorithm)**:
   - Explanation: Finds the maximum sum of a contiguous subarray using Kadane’s algorithm.

7. **Merge Two Sorted Arrays**:
   - Explanation: Merge two sorted arrays into one sorted array.

8. **Valid Parentheses**:
   - Explanation: Check if a string of parentheses is valid using a stack.

9. **Find Duplicates in an Array**:
   - Explanation: Find all duplicates in an array where integers are in the range from 1 to N.

10. **Climbing Stairs (DP)**:
   - Explanation: You are climbing a staircase with N steps. You can climb 1 or 2 steps at a time. Calculate how many distinct ways you can reach the top.

11. **Reverse a Linked List (Iterative)**:
   - Explanation: Reverse a singly linked list iteratively by changing the next pointers.

12. **Rotate Array (Reversal Algorithm)**:
   - Explanation: Rotate an array to the right by k steps by reversing parts of the array.

13. **Majority Element (Boyer-Moore Voting Algorithm)**:
   - Explanation: Find the majority element that appears more than n/2 times in the array.

14. **Merge Intervals**:
   - Explanation: Merge overlapping intervals.

15. **Detect Cycle in a Linked List (Floyd's Cycle Detection)**:
   - Explanation: Use two pointers to detect a cycle in a linked list.

16. **Product of Array Except Self**:
   - Explanation: Compute an array where each element is the product of all elements except itself.

17. **Longest Common Prefix**:
   - Explanation: Find the longest common prefix among a list of strings.

18. **Flatten Binary Tree to Linked List**:
   - Explanation: Flatten a binary tree to a "linked list" in-place.

19. **Word Search (Backtracking)**:
   - Explanation: Given a board and a word, check if the word exists in the grid using DFS.

20. **First Missing Positive**:
   - Explanation: Find the smallest missing positive integer in an unsorted array.

21. **Counting Bits**:
   - Explanation: For each number from 0 to n, compute the number of 1's in their binary representation.

22. **Set Matrix Zeroes**:
   - Explanation: Given an MxN matrix, set entire row and column to zeroes if an element is zero.

23. **Binary Tree Level Order Traversal**:
   - Explanation: Traverse a binary tree level by level, storing each level as a list.

24. **Find Peak Element**:
   - Explanation: Find a peak element in an array such that the element is greater than its neighbors.

25. **Permutations (Backtracking)**:
   - Explanation: Generate all possible permutations of a list of numbers.

26. **Combination Sum (Backtracking)**:
   - Explanation: Given a set of candidate numbers, find all unique combinations that sum to a target.

27. **Letter Combinations of a Phone Number (Backtracking)**:
   - Explanation: Given a string of digits, return all possible letter combinations that the digits can represent.

28. **Valid Anagram**:
   - Explanation: Check if two strings are anagrams of each other by comparing character counts.

29. **Minimum Depth of Binary Tree**:
   - Explanation: Find the minimum depth of a binary tree using BFS.

30. **Merge K Sorted Lists (Priority Queue)**:
   - Explanation: Merge k sorted linked lists into one sorted list using a priority queue.

31. **LCA of Binary Search Tree**:
   - Explanation: Find the lowest common ancestor (LCA) of two nodes in a Binary Search Tree.

32. **Search in Rotated Sorted Array**:
   - Explanation: Search for a target value in a rotated sorted array using binary search.

33. **Coin Change (Dynamic Programming)**:
   - Explanation: Given a set of coins, determine the minimum number of coins needed to make a certain amount.

34. **Jump Game (Greedy)**:
   - Explanation: Given an array of non-negative integers, determine if you can reach the last index.

35. **Kth Largest Element in an Array (Quickselect)**:
   - Explanation: Find the k-th largest element in an unsorted array using the quickselect algorithm.

36. **House Robber (Dynamic Programming)**:
   - Explanation: Solve the problem of maximizing the amount of money you can rob without robbing two adjacent houses.

37. **Longest Palindromic Substring (Expand Around Center)**:
   - Explanation: Find the longest palindromic substring by expanding around each possible center.

38. **Subsets (Backtracking)**:
   - Explanation: Generate all possible subsets of a list of numbers.

39. **Rotate Image (In-Place)**:
   - Explanation: Rotate an NxN 2D matrix by 90 degrees in place.

40. **Unique Paths (Dynamic Programming)**:
   - Explanation: Find the number of unique paths from the top-left to the bottom-right of an m x n grid.

41. **Binary Tree Maximum Path Sum (DFS)**:
   - Explanation: Find the maximum path sum in a binary tree where the path can start and end at any node.

42. **Word Break (Dynamic Programming)**:
   - Explanation: Given a string and a dictionary of words, determine if the string can be segmented into a space-separated sequence of dictionary words.

43. **Reorder List**:
   - Explanation: Given a singly linked list, reorder it so that nodes are arranged in a specific pattern.

44. **Longest Consecutive Sequence**:
   - Explanation: Given an unsorted array, find the length of the longest consecutive elements sequence.

45. **Zigzag Level Order Traversal**:
   - Explanation: Perform a level-order traversal of a binary tree where nodes at each level are traversed in zigzag order.

46. **Longest Substring Without Repeating Characters**:
   - Explanation: Find the length of the longest substring without repeating characters.

47. **Spiral Matrix**:
   - Explanation: Given a matrix, return all elements of the matrix in spiral order.

48. **Alien Dictionary (Topological Sort)**:
   - Explanation: Given a sorted dictionary of an alien language, derive the order of characters in the alien alphabet.

49. **Valid Sudoku**:
   - Explanation: Determine if a 9x9 Sudoku board is valid according to the Sudoku rules.

50. **Decode Ways (Dynamic Programming)**:
   - Explanation: Given a string of digits, determine the total number of ways to decode it.

51. **Kth Smallest Element in a Sorted Matrix**:
   - Explanation: Find the k-th smallest element in a sorted matrix using a priority queue.

52. **Trapping Rain Water**:
   - Explanation: Calculate how much water can be trapped after raining on an elevation map.

53. **Word Ladder (BFS)**:
   - Explanation: Find the shortest transformation sequence from a word to another using BFS.

54. **Edit Distance (Dynamic Programming)**:
   - Explanation: Find the minimum number of operations to convert one string into another.

55. **Longest Increasing Path in a Matrix**:
   - Explanation: Find the longest increasing path in a matrix using DFS and memoization.

56. **Course Schedule II (Topological Sort)**:
   - Explanation: Find the order of courses to take given their prerequisites using topological sorting.

57. **Maximum Product Subarray**:
   - Explanation: Find the contiguous subarray with the largest product in an array.

58. **Longest Palindromic Subsequence (Dynamic Programming)**:
   - Explanation: Find the longest palindromic subsequence in a given string.

59. **N-Queens Problem (Backtracking)**:
   - Explanation: Solve the N-Queens problem and return all distinct solutions.

60. **Sudoku Solver (Backtracking)**:
   - Explanation: Solve a Sudoku puzzle by filling in the empty cells using backtracking.
"""






"""
1. **Two Sum** (HashMap Approach):
- **Explanation**: Given an array of integers and a target sum, find two numbers in the array that add up to the target.
- **Time Complexity**: O(N), since we traverse the array once.
- **Space Complexity**: O(N), for storing the numbers in a hash map.
"""
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

"""
2. **Palindrome Check** (Two Pointer Approach):
- **Explanation**: Check if a given string is a palindrome by comparing characters from both ends.
- **Time Complexity**: O(N), where N is the length of the string.
- **Space Complexity**: O(1), constant space.
"""
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True

"""
3. **Fibonacci Sequence** (Dynamic Programming):
- **Explanation**: Compute the N-th Fibonacci number using DP to avoid recomputation.
- **Time Complexity**: O(N), where N is the desired Fibonacci number.
- **Space Complexity**: O(1), since we only need two variables to store the last two Fibonacci numbers.
"""
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

"""
4. **Find the Missing Number** (XOR Approach):
- **Explanation**: Given an array of numbers from 1 to N, find the missing number.
- **Time Complexity**: O(N), single traversal through the array.
- **Space Complexity**: O(1), no extra space needed.
"""
def find_missing_number(arr, n):
    xor_sum = 0
    for i in range(1, n + 1):
        xor_sum ^= i
    for num in arr:
        xor_sum ^= num
    return xor_sum

"""
5. **Binary Search** (Iterative):
- **Explanation**: Perform a binary search on a sorted array to find a target value.
- **Time Complexity**: O(log N), where N is the size of the array.
- **Space Complexity**: O(1), as no extra space is needed.
"""
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

"""
6. **Maximum Subarray Sum (Kadane's Algorithm)**:
- **Explanation**: Finds the maximum sum of a contiguous subarray using Kadane’s algorithm.
- **Time Complexity**: O(N), where N is the number of elements in the array.
- **Space Complexity**: O(1), constant space used.
"""
def max_subarray_sum(arr):
    max_ending_here = max_so_far = arr[0]
    for num in arr[1:]:
        max_ending_here = max(num, max_ending_here + num)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far

"""
7. **Merge Two Sorted Arrays**:
- **Explanation**: Merge two sorted arrays into one sorted array.
- **Time Complexity**: O(N + M), where N and M are the lengths of the two arrays.
- **Space Complexity**: O(N + M), for the result array.
"""
def merge_sorted_arrays(arr1, arr2):
    merged = []
    i = j = 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            merged.append(arr1[i])
            i += 1
        else:
            merged.append(arr2[j])
            j += 1
    merged.extend(arr1[i:])
    merged.extend(arr2[j:])
    return merged

"""
8. **Valid Parentheses**:
- **Explanation**: Check if a string of parentheses is valid using a stack.
- **Time Complexity**: O(N), where N is the length of the string.
- **Space Complexity**: O(N), due to the stack.
"""
def is_valid_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)
    return not stack

"""
9. **Find Duplicates in an Array**:
- **Explanation**: Find all duplicates in an array where integers are in the range from 1 to N.
- **Time Complexity**: O(N), where N is the number of elements in the array.
- **Space Complexity**: O(1), constant space by using negative marking technique.
"""
def find_duplicates(nums):
    res = []
    for num in nums:
        if nums[abs(num) - 1] < 0:
            res.append(abs(num))
        else:
            nums[abs(num) - 1] *= -1
    return res

"""
10. **Climbing Stairs (DP)**:
- **Explanation**: You are climbing a staircase with N steps. You can climb 1 or 2 steps at a time. Calculate how many distinct ways you can reach the top.
- **Time Complexity**: O(N), where N is the number of steps.
- **Space Complexity**: O(1), constant space to track only the last two ways.
"""
def climb_stairs(n):
    if n == 1:
        return 1
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b


"""
11. **Reverse a Linked List** (Iterative):
- **Explanation**: Reverse a singly linked list iteratively by changing the next pointers.
- **Time Complexity**: O(N), where N is the number of nodes in the list.
- **Space Complexity**: O(1), constant space.
"""
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev

"""
12. **Rotate Array** (Reversal Algorithm):
- **Explanation**: Rotate an array to the right by k steps by reversing parts of the array.
- **Time Complexity**: O(N), where N is the length of the array.
- **Space Complexity**: O(1), since no extra space is used.
"""
def rotate_array(nums, k):
    k %= len(nums)
    nums.reverse()
    nums[:k] = reversed(nums[:k])
    nums[k:] = reversed(nums[k:])

"""
13. **Majority Element** (Boyer-Moore Voting Algorithm):
- **Explanation**: Find the majority element that appears more than n/2 times in the array.
- **Time Complexity**: O(N), where N is the number of elements.
- **Space Complexity**: O(1), constant space used.
"""
def majority_element(nums):
    count = 0
    candidate = None
    for num in nums:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
    return candidate

"""
14. **Merge Intervals**:
- **Explanation**: Merge overlapping intervals.
- **Time Complexity**: O(N log N), where N is the number of intervals (sorting dominates).
- **Space Complexity**: O(N), for storing the result.
"""
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged

"""
15. **Detect Cycle in a Linked List** (Floyd's Cycle Detection):
- **Explanation**: Use two pointers to detect a cycle in a linked list.
- **Time Complexity**: O(N), where N is the number of nodes.
- **Space Complexity**: O(1), constant space.
"""
def has_cycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

"""
16. **Product of Array Except Self**:
- **Explanation**: Compute an array where each element is the product of all elements except itself.
- **Time Complexity**: O(N), where N is the number of elements.
- **Space Complexity**: O(1), since the output array does not count as extra space.
"""
def product_except_self(nums):
    length = len(nums)
    answer = [1] * length
    left_product = 1
    for i in range(length):
        answer[i] = left_product
        left_product *= nums[i]
    right_product = 1
    for i in range(length - 1, -1, -1):
        answer[i] *= right_product
        right_product *= nums[i]
    return answer

"""
17. **Longest Common Prefix**:
- **Explanation**: Find the longest common prefix among a list of strings.
- **Time Complexity**: O(S), where S is the sum of all characters in the strings.
- **Space Complexity**: O(1), constant space.
"""
def longest_common_prefix(strs):
    if not strs:
        return ""
    prefix = strs[0]
    for string in strs[1:]:
        while string.find(prefix) != 0:
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix

"""
18. **Flatten Binary Tree to Linked List**:
- **Explanation**: Flatten a binary tree to a "linked list" in-place.
- **Time Complexity**: O(N), where N is the number of nodes.
- **Space Complexity**: O(N), due to recursion stack.
"""
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def flatten_tree(root):
    def flatten(node):
        if not node:
            return None
        left_tail = flatten(node.left)
        right_tail = flatten(node.right)
        if left_tail:
            left_tail.right = node.right
            node.right = node.left
            node.left = None
        return right_tail or left_tail or node
    flatten(root)

"""
19. **Word Search (Backtracking)**:
- **Explanation**: Given a board and a word, check if the word exists in the grid using DFS.
- **Time Complexity**: O(M * N * 4^L), where M is rows, N is columns, and L is the length of the word.
- **Space Complexity**: O(L), recursion stack of depth L.
"""
def exist(board, word):
    def dfs(i, j, k):
        if k == len(word):
            return True
        if i < 0 or j < 0 or i >= len(board) or j >= len(board[0]) or board[i][j] != word[k]:
            return False
        temp, board[i][j] = board[i][j], '#'
        found = dfs(i+1, j, k+1) or dfs(i-1, j, k+1) or dfs(i, j+1, k+1) or dfs(i, j-1, k+1)
        board[i][j] = temp
        return found
    for i in range(len(board)):
        for j in range(len(board[0])):
            if dfs(i, j, 0):
                return True
    return False

"""
20. **First Missing Positive**:
- **Explanation**: Find the smallest missing positive integer in an unsorted array.
- **Time Complexity**: O(N), where N is the length of the array.
- **Space Complexity**: O(1), no extra space used.
"""
def first_missing_positive(nums):
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1

"""
21. **Counting Bits**:
- **Explanation**: For each number from 0 to n, compute the number of 1's in their binary representation.
- **Time Complexity**: O(N), where N is the given number.
- **Space Complexity**: O(N), for the result array.
"""
def count_bits(n):
    result = [0] * (n + 1)
    for i in range(1, n + 1):
        result[i] = result[i >> 1] + (i & 1)
    return result

"""
22. **Set Matrix Zeroes**:
- **Explanation**: Given an MxN matrix, set entire row and column to zeroes if an element is zero.
- **Time Complexity**: O(M * N), where M is the number of rows and N is the number of columns.
- **Space Complexity**: O(1), since we modify the matrix in-place.
"""
def set_zeroes(matrix):
    rows, cols = len(matrix), len(matrix[0])
    row_zero, col_zero = False, False
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 0:
                matrix[0][j] = 0
                matrix[i][0] = 0
                if i == 0:
                    row_zero = True
                if j == 0:
                    col_zero = True
    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    if row_zero:
        for j in range(cols):
            matrix[0][j] = 0
    if col_zero:
        for i in range(rows):
            matrix[i][0] = 0

"""
23. **Binary Tree Level Order Traversal**:
- **Explanation**: Traverse a binary tree level by level, storing each level as a list.
- **Time Complexity**: O(N), where N is the number of nodes.
- **Space Complexity**: O(N), for storing the levels.
"""
def level_order(root):
    result = []
    if not root:
        return result
    queue = [root]
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.pop(0)
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result

"""
24. **Find Peak Element**:
- **Explanation**: Find a peak element in an array such that the element is greater than its neighbors.
- **Time Complexity**: O(log N), using binary search.
- **Space Complexity**: O(1), constant space.
"""
def find_peak_element(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < nums[mid + 1]:
            left = mid + 1
        else:
            right = mid
    return left

"""
25. **Permutations** (Backtracking):
- **Explanation**: Generate all possible permutations of a list of numbers.
- **Time Complexity**: O(N!), where N is the number of elements.
- **Space Complexity**: O(N!), for storing all permutations.
"""
def permute(nums):
    result = []
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]
    backtrack(0)
    return result

"""
26. **Combination Sum** (Backtracking):
- **Explanation**: Given a set of candidate numbers, find all unique combinations that sum to a target.
- **Time Complexity**: O(2^N), where N is the number of candidates.
- **Space Complexity**: O(T), where T is the target sum.
"""
def combination_sum(candidates, target):
    result = []
    def backtrack(start, path, total):
        if total == target:
            result.append(path)
            return
        if total > target:
            return
        for i in range(start, len(candidates)):
            backtrack(i, path + [candidates[i]], total + candidates[i])
    backtrack(0, [], 0)
    return result

"""
27. **Letter Combinations of a Phone Number** (Backtracking):
- **Explanation**: Given a string of digits, return all possible letter combinations that the digits can represent.
- **Time Complexity**: O(4^N), where N is the length of the input digits (since each digit can map to up to 4 letters).
- **Space Complexity**: O(N), for recursion depth.
"""
def letter_combinations(digits):
    if not digits:
        return []
    phone = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
    result = []
    def backtrack(combo, next_digits):
        if len(next_digits) == 0:
            result.append(combo)
        else:
            for letter in phone[next_digits[0]]:
                backtrack(combo + letter, next_digits[1:])
    backtrack("", digits)
    return result

"""
28. **Valid Anagram**:
- **Explanation**: Check if two strings are anagrams of each other by comparing character counts.
- **Time Complexity**: O(N), where N is the length of the strings.
- **Space Complexity**: O(1), constant space assuming the character set is fixed (e.g., lowercase English letters).
"""
def is_anagram(s, t):
    return sorted(s) == sorted(t)

"""
29. **Minimum Depth of Binary Tree**:
- **Explanation**: Find the minimum depth of a binary tree using BFS.
- **Time Complexity**: O(N), where N is the number of nodes.
- **Space Complexity**: O(N), for the queue.
"""
def min_depth(root):
    if not root:
        return 0
    queue = [(root, 1)]
    while queue:
        node, depth = queue.pop(0)
        if not node.left and not node.right:
            return depth
        if node.left:
            queue.append((node.left, depth + 1))
        if node.right:
            queue.append((node.right, depth + 1))

"""
30. **Merge K Sorted Lists** (Priority Queue):
- **Explanation**: Merge k sorted linked lists into one sorted list using a priority queue.
- **Time Complexity**: O(N log k), where N is the total number of elements and k is the number of linked lists.
- **Space Complexity**: O(k), for storing the elements in the priority queue.
"""
import heapq
def merge_k_lists(lists):
    heap = []
    for l in lists:
        while l:
            heapq.heappush(heap, l.val)
            l = l.next
    dummy = ListNode(0)
    curr = dummy
    while heap:
        curr.next = ListNode(heapq.heappop(heap))
        curr = curr.next
    return dummy.next

"""
31. **LCA of Binary Search Tree**:
- **Explanation**: Find the lowest common ancestor (LCA) of two nodes in a Binary Search Tree.
- **Time Complexity**: O(H), where H is the height of the tree.
- **Space Complexity**: O(1), since no extra space is used.
"""
def lowest_common_ancestor(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root

"""
32. **Search in Rotated Sorted Array**:
- **Explanation**: Search for a target value in a rotated sorted array using binary search.
- **Time Complexity**: O(log N), where N is the number of elements.
- **Space Complexity**: O(1), no extra space is used.
"""
def search_rotated_array(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1

"""
33. **Coin Change** (Dynamic Programming):
- **Explanation**: Given a set of coins, determine the minimum number of coins needed to make a certain amount.
- **Time Complexity**: O(N * M), where N is the amount and M is the number of coin denominations.
- **Space Complexity**: O(N), for storing the DP array.
"""
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

"""
34. **Jump Game** (Greedy):
- **Explanation**: Given an array of non-negative integers, determine if you can reach the last index.
- **Time Complexity**: O(N), where N is the number of elements.
- **Space Complexity**: O(1), since we use only a few variables.
"""
def can_jump(nums):
    max_reach = 0
    for i, num in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + num)
    return True

"""
35. **Kth Largest Element in an Array** (Quickselect):
- **Explanation**: Find the k-th largest element in an unsorted array using the quickselect algorithm.
- **Time Complexity**: O(N) on average, where N is the number of elements.
- **Space Complexity**: O(1), since we do not use extra space.
"""
def partition(nums, left, right):
    pivot = nums[right]
    i = left
    for j in range(left, right):
        if nums[j] <= pivot:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
    nums[i], nums[right] = nums[right], nums[i]
    return i

def quickselect(nums, left, right, k):
    if left == right:
        return nums[left]
    pivot_index = partition(nums, left, right)
    if k == pivot_index:
        return nums[k]
    elif k < pivot_index:
        return quickselect(nums, left, pivot_index - 1, k)
    else:
        return quickselect(nums, pivot_index + 1, right, k)

def find_kth_largest(nums, k):
    return quickselect(nums, 0, len(nums) - 1, len(nums) - k)

"""
36. **House Robber** (Dynamic Programming):
- **Explanation**: Solve the problem of maximizing the amount of money you can rob without robbing two adjacent houses.
- **Time Complexity**: O(N), where N is the number of houses.
- **Space Complexity**: O(1), since we only use two variables to store results.
"""
def rob(nums):
    prev1, prev2 = 0, 0
    for num in nums:
        temp = prev1
        prev1 = max(prev2 + num, prev1)
        prev2 = temp
    return prev1

"""
37. **Longest Palindromic Substring** (Expand Around Center):
- **Explanation**: Find the longest palindromic substring by expanding around each possible center.
- **Time Complexity**: O(N^2), where N is the length of the string.
- **Space Complexity**: O(1), since no extra space is used.
"""
def longest_palindrome(s):
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]
    
    if len(s) == 0:
        return ""
    
    longest = ""
    for i in range(len(s)):
        odd_pal = expand_around_center(i, i)
        even_pal = expand_around_center(i, i + 1)
        longest = max(longest, odd_pal, even_pal, key=len)
    
    return longest

"""
38. **Subsets** (Backtracking):
- **Explanation**: Generate all possible subsets of a list of numbers.
- **Time Complexity**: O(2^N), where N is the number of elements.
- **Space Complexity**: O(N), for recursion depth.
"""
def subsets(nums):
    result = []
    def backtrack(start, path):
        result.append(path)
        for i in range(start, len(nums)):
            backtrack(i + 1, path + [nums[i]])
    backtrack(0, [])
    return result

"""
39. **Rotate Image (In-Place)**:
- **Explanation**: Rotate an NxN 2D matrix by 90 degrees in place.
- **Time Complexity**: O(N^2), where N is the size of the matrix.
- **Space Complexity**: O(1), as it is in-place.
"""
def rotate(matrix):
    matrix.reverse()
    for i in range(len(matrix)):
        for j in range(i):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

"""
40. **Unique Paths (Dynamic Programming)**:
- **Explanation**: Find the number of unique paths from the top-left to the bottom-right of an m x n grid.
- **Time Complexity**: O(M * N), where M is the number of rows and N is the number of columns.
- **Space Complexity**: O(M * N), for storing the DP table.
"""
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]

"""
41. **Binary Tree Maximum Path Sum** (DFS):
- **Explanation**: Find the maximum path sum in a binary tree where the path can start and end at any node.
- **Time Complexity**: O(N), where N is the number of nodes.
- **Space Complexity**: O(H), where H is the height of the tree (recursion depth).
"""
def max_path_sum(root):
    def dfs(node):
        if not node:
            return 0
        # Recursively compute the maximum sum on the left and right, and ignore negatives
        left = max(dfs(node.left), 0)
        right = max(dfs(node.right), 0)
        # Calculate the current path sum with the current node as the root
        current_max = node.val + left + right
        # Update global maximum path sum
        nonlocal max_sum
        max_sum = max(max_sum, current_max)
        # Return the maximum gain to continue the path
        return node.val + max(left, right)
    
    max_sum = float('-inf')
    dfs(root)
    return max_sum


"""
42. **Word Break (Dynamic Programming)**:
- **Explanation**: Given a string and a dictionary of words, determine if the string can be segmented into a space-separated sequence of dictionary words.
- **Time Complexity**: O(N^2), where N is the length of the string.
- **Space Complexity**: O(N), for the DP array.
"""
def word_break(s, word_dict):
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_dict:
                dp[i] = True
                break
    return dp[len(s)]

"""
43. **Reorder List**:
- **Explanation**: Given a singly linked list, reorder it so that nodes are arranged in the following pattern: L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → …
- **Time Complexity**: O(N), where N is the number of nodes.
- **Space Complexity**: O(1), since the list is modified in-place.
"""
def reorder_list(head):
    if not head:
        return
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    second = slow.next
    slow.next = None
    prev = None
    while second:
        next_node = second.next
        second.next = prev
        prev = second
        second = next_node
    first, second = head, prev
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first = tmp1
        second = tmp2

"""
44. **Longest Consecutive Sequence**:
- **Explanation**: Given an unsorted array, find the length of the longest consecutive elements sequence.
- **Time Complexity**: O(N), where N is the number of elements.
- **Space Complexity**: O(N), for storing elements in a set.
"""
def longest_consecutive(nums):
    num_set = set(nums)
    longest_streak = 0
    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1
            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1
            longest_streak = max(longest_streak, current_streak)
    return longest_streak

"""
45. **Zigzag Level Order Traversal**:
- **Explanation**: Perform a level-order traversal of a binary tree where nodes at each level are traversed in zigzag order.
- **Time Complexity**: O(N), where N is the number of nodes.
- **Space Complexity**: O(N), for storing nodes in the queue.
"""
def zigzag_level_order(root):
    if not root:
        return []
    result = []
    queue = [root]
    left_to_right = True
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.pop(0)
            if left_to_right:
                level.append(node.val)
            else:
                level.insert(0, node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
        left_to_right = not left_to_right
    return result

"""
46. **Longest Substring Without Repeating Characters**:
- **Explanation**: Find the length of the longest substring without repeating characters.
- **Time Complexity**: O(N), where N is the length of the string.
- **Space Complexity**: O(min(N, M)), where M is the size of the character set.
"""
def length_of_longest_substring(s):
    char_map = {}
    left = 0
    max_len = 0
    for right in range(len(s)):
        if s[right] in char_map:
            left = max(left, char_map[s[right]] + 1)
        char_map[s[right]] = right
        max_len = max(max_len, right - left + 1)
    return max_len

"""
47. **Spiral Matrix**:
- **Explanation**: Given a matrix, return all elements of the matrix in spiral order.
- **Time Complexity**: O(N), where N is the total number of elements in the matrix.
- **Space Complexity**: O(1), for the output array.
"""
def spiral_order(matrix):
    result = []
    while matrix:
        result += matrix.pop(0)
        matrix = list(zip(*matrix))[::-1]
    return result

"""
48. **Alien Dictionary** (Topological Sort):
- **Explanation**: Given a sorted dictionary of an alien language, derive the order of characters in the alien alphabet.
- **Time Complexity**: O(C + N), where C is the total length of all words and N is the number of characters.
- **Space Complexity**: O(N), for storing the graph.
"""
from collections import defaultdict, deque

def alien_order(words):
    graph = defaultdict(set)
    in_degree = {char: 0 for word in words for char in word}
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        for c1, c2 in zip(word1, word2):
            if c1 != c2:
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break
    queue = deque([char for char in in_degree if in_degree[char] == 0])
    result = []
    while queue:
        char = queue.popleft()
        result.append(char)
        for neighbor in graph[char]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    if len(result) < len(in_degree):
        return ""
    return "".join(result)

"""
49. **Valid Sudoku**:
- **Explanation**: Determine if a 9x9 Sudoku board is valid according to the Sudoku rules.
- **Time Complexity**: O(1), since the board size is fixed.
- **Space Complexity**: O(1), since no extra space is used.
"""
def is_valid_sudoku(board):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    for r in range(9):
        for c in range(9):
            if board[r][c] == '.':
                continue
            if board[r][c] in rows[r] or board[r][c] in cols[c] or board[r][c] in boxes[(r // 3) * 3 + (c // 3)]:
                return False
            rows[r].add(board[r][c])
            cols[c].add(board[r][c])
            boxes[(r // 3) * 3 + (c // 3)].add(board[r][c])
    return True

"""
50. **Decode Ways (Dynamic Programming)**:
- **Explanation**: Given a string of digits, determine the total number of ways to decode it.
- **Time Complexity**: O(N), where N is the length of the string.
- **Space Complexity**: O(N), for the DP array.
"""
def num_decodings(s):
    if not s or s[0] == '0':
        return 0
    dp = [0] * (len(s) + 1)
    dp[0], dp[1] = 1, 1
    for i in range(2, len(s) + 1):
        if s[i - 1] != '0':
            dp[i] += dp[i - 1]
        if 10 <= int(s[i - 2:i]) <= 26:
            dp[i] += dp[i - 2]
    return dp[-1]


"""
1. **Kth Smallest Element in a Sorted Matrix**:
- **Explanation**: Given an N x N matrix where each row and column is sorted, find the k-th smallest element.
- **Time Complexity**: O(k log N), where N is the size of the matrix and k is the index of the element.
- **Space Complexity**: O(N), for storing elements in a priority queue.
"""
import heapq
def kth_smallest(matrix, k):
    n = len(matrix)
    min_heap = [(matrix[i][0], i, 0) for i in range(n)]
    heapq.heapify(min_heap)
    for _ in range(k - 1):
        val, r, c = heapq.heappop(min_heap)
        if c + 1 < n:
            heapq.heappush(min_heap, (matrix[r][c + 1], r, c + 1))
    return heapq.heappop(min_heap)[0]

"""
2. **Trapping Rain Water**:
- **Explanation**: Given an array representing the elevation map, calculate how much water it can trap after raining.
- **Time Complexity**: O(N), where N is the number of elements.
- **Space Complexity**: O(1), constant space.
"""
def trap(height):
    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    water = 0
    while left < right:
        if height[left] < height[right]:
            left_max = max(left_max, height[left])
            water += left_max - height[left]
            left += 1
        else:
            right_max = max(right_max, height[right])
            water += right_max - height[right]
            right -= 1
    return water

"""
3. **Word Ladder (BFS)**:
- **Explanation**: Find the length of the shortest transformation sequence from beginWord to endWord using BFS.
- **Time Complexity**: O(N * M), where N is the number of words and M is the length of each word.
- **Space Complexity**: O(N), for storing words in the queue.
"""
from collections import deque
def word_ladder(begin_word, end_word, word_list):
    word_list = set(word_list)
    if end_word not in word_list:
        return 0
    queue = deque([(begin_word, 1)])
    while queue:
        word, length = queue.popleft()
        if word == end_word:
            return length
        for i in range(len(word)):
            for char in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + char + word[i + 1:]
                if next_word in word_list:
                    word_list.remove(next_word)
                    queue.append((next_word, length + 1))
    return 0

"""
4. **Edit Distance (Dynamic Programming)**:
- **Explanation**: Given two strings, find the minimum number of operations (insert, delete, replace) to convert one string to another.
- **Time Complexity**: O(M * N), where M and N are the lengths of the two strings.
- **Space Complexity**: O(M * N), for storing the DP table.
"""
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
    return dp[m][n]

"""
5. **Longest Increasing Path in a Matrix**:
- **Explanation**: Given a matrix, find the length of the longest increasing path in the matrix.
- **Time Complexity**: O(M * N), where M is the number of rows and N is the number of columns.
- **Space Complexity**: O(M * N), for the memoization table.
"""
def longest_increasing_path(matrix):
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    memo = [[-1] * n for _ in range(m)]
    
    def dfs(x, y):
        if memo[x][y] != -1:
            return memo[x][y]
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        longest = 1
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] > matrix[x][y]:
                longest = max(longest, 1 + dfs(nx, ny))
        memo[x][y] = longest
        return longest
    
    return max(dfs(x, y) for x in range(m) for y in range(n))

"""
6. **Course Schedule II (Topological Sort)**:
- **Explanation**: Find the order of courses to take given their prerequisites using topological sorting.
- **Time Complexity**: O(V + E), where V is the number of courses and E is the number of prerequisites.
- **Space Complexity**: O(V + E), for storing the graph.
"""
from collections import deque
def find_order(num_courses, prerequisites):
    graph = {i: [] for i in range(num_courses)}
    in_degree = [0] * num_courses
    for dest, src in prerequisites:
        graph[src].append(dest)
        in_degree[dest] += 1
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    order = []
    while queue:
        course = queue.popleft()
        order.append(course)
        for neighbor in graph[course]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return order if len(order) == num_courses else []

"""
7. **Maximum Product Subarray**:
- **Explanation**: Given an integer array, find the contiguous subarray that has the largest product.
- **Time Complexity**: O(N), where N is the number of elements.
- **Space Complexity**: O(1), constant space.
"""
def max_product(nums):
    max_prod = min_prod = result = nums[0]
    for i in range(1, len(nums)):
        if nums[i] < 0:
            max_prod, min_prod = min_prod, max_prod
        max_prod = max(nums[i], max_prod * nums[i])
        min_prod = min(nums[i], min_prod * nums[i])
        result = max(result, max_prod)
    return result

"""
8. **Longest Palindromic Subsequence (Dynamic Programming)**:
- **Explanation**: Find the longest palindromic subsequence in a given string.
- **Time Complexity**: O(N^2), where N is the length of the string.
- **Space Complexity**: O(N^2), for storing the DP table.
"""
def longest_palindromic_subsequence(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]

"""
9. **N-Queens Problem (Backtracking)**:
- **Explanation**: Solve the N-Queens problem and return all possible distinct solutions.
- **Time Complexity**: O(N!), where N is the size of the board.
- **Space Complexity**: O(N), for storing the board.
"""
def solve_n_queens(n):
    result = []
    board = [["."] * n for _ in range(n)]
    
    def is_valid(row, col):
        for i in range(row):
            if board[i][col] == "Q":
                return False
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == "Q":
                return False
        for i, j in zip(range(row, -1, -1), range(col, n)):
            if board[i][j] == "Q":
                return False
        return True
    
    def backtrack(row):
        if row == n:
            result.append(["".join(r) for r in board])
            return
        for col in range(n):
            if is_valid(row, col):
                board[row][col] = "Q"
                backtrack(row + 1)
                board[row][col] = "."
    
    backtrack(0)
    return result

"""
10. **Sudoku Solver (Backtracking)**:
- **Explanation**: Solve a Sudoku puzzle by filling in the empty cells.
- **Time Complexity**: O(9^N), where N is the number of empty cells.
- **Space Complexity**: O(N), for storing the board state.
"""
def solve_sudoku(board):
    def is_valid(board, row, col, num):
        for i in range(9):
            if board[i][col] == num or board[row][i] == num:
                return False
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if board[start_row + i][start_col + j] == num:
                    return False
        return True
    
    def backtrack():
        for i in range(9):
            for j in range(9):
                if board[i][j] == ".":
                    for num in map(str, range(1, 10)):
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            if backtrack():
                                return True
                            board[i][j] = "."
                    return False
        return True
    
    backtrack()
