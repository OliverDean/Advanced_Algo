"""
    Huffman Codes: Long Description

Huffman coding is a popular greedy algorithm used for lossless data compression. It creates a variable-length code for each symbol in the input based on their frequencies. Symbols that occur more frequently are assigned shorter codes, while those that occur less frequently are given longer codes. The result is a compressed encoding that minimizes the total number of bits required to represent the input.

The key concept in Huffman coding is to represent symbols as leaves in a binary tree. The more frequent a symbol, the closer it is to the root, and thus it gets a shorter binary code. This is done in a way that the code for each symbol is prefix-free—no code is a prefix of another. This ensures that the encoded message can be uniquely decoded.

The steps of the Huffman coding algorithm are as follows:

    Count the Frequency: Compute the frequency of each symbol in the input.
    Build a Priority Queue: Insert each symbol as a leaf node into a priority queue (min-heap), with the priority determined by the symbol's frequency.
    Build the Huffman Tree: Repeatedly remove the two nodes with the smallest frequency from the queue and create a new parent node with a combined frequency. Insert the parent node back into the queue.
    Generate Codes: Once the tree is complete, generate the Huffman code for each symbol by tracing the path from the root to the corresponding leaf. A left traversal represents a '0', and a right traversal represents a '1'.
    Encode the Input: Use the generated codes to encode the input data.

Huffman coding is optimal in terms of minimizing the expected length of the encoded message when symbol frequencies are known.
"""

class Node:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

def insert_in_order(queue, node):
    """
    Inserts a node into the priority queue (list) in ascending order based on frequency.
    This maintains the "min-heap" like behavior without using the heapq library.
    
    Time Complexity: O(n) where n is the number of elements in the queue.
    """
    index = 0
    while index < len(queue) and queue[index].freq < node.freq:
        index += 1
    queue.insert(index, node)

def huffman_encoding(frequencies):
    """
    This function performs Huffman coding on the input frequencies of characters.
    
    Steps:
    1. Build a priority queue (min-heap) from the characters and their frequencies manually.
    2. Build a binary tree by combining the two lowest frequency nodes repeatedly.
    3. Assign binary codes based on the tree structure: left = '0', right = '1'.
    
    Time Complexity:
    - O(n^2), where n is the number of unique characters. This is because we insert elements manually in sorted order.

    Space Complexity:
    - O(n), where n is the number of unique characters, for storing the tree and the codes.

    Parameters:
    - frequencies: A dictionary where keys are characters and values are their frequencies.

    Returns:
    - A dictionary where each character is associated with its Huffman code.
    """
    
    # Create a priority queue initialized with leaf nodes
    queue = []
    for char, freq in frequencies.items():
        insert_in_order(queue, Node(char, freq))

    # Build the Huffman tree
    while len(queue) > 1:
        # Remove the two nodes with the smallest frequencies
        left = queue.pop(0)
        right = queue.pop(0)

        # Create a new parent node with combined frequency
        merged = Node(None, left.freq + right.freq, left, right)
        insert_in_order(queue, merged)

    # The root node of the Huffman tree
    root = queue[0]

    # Recursive function to assign codes to characters
    def generate_codes(node, current_code, codes):
        if node is None:
            return
        if node.char is not None:
            codes[node.char] = current_code
            return
        generate_codes(node.left, current_code + "0", codes)
        generate_codes(node.right, current_code + "1", codes)

    # Generate the Huffman codes
    codes = {}
    generate_codes(root, "", codes)
    return codes

# Example usage
def huffman_example():
    input_string = "streets are stone stars are hot"
    frequency_counter = {}
    
    # Manually count frequencies
    for char in input_string:
        if char in frequency_counter:
            frequency_counter[char] += 1
        else:
            frequency_counter[char] = 1

    huffman_codes = huffman_encoding(frequency_counter)

    print("Huffman Codes for each character:")
    for char, code in huffman_codes.items():
        print(f"'{char}': {code}")

    # Encoding the input string
    encoded_string = ''.join(huffman_codes[char] for char in input_string)
    print(f"\nEncoded string: {encoded_string}")
    print(f"Original length: {len(input_string) * 8} bits")
    print(f"Encoded length: {len(encoded_string)} bits")

huffman_example()
