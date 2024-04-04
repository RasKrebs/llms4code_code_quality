from heapq import heappop, heappush
from collections import defaultdict
import string

class Node(object):
    def __init__(self, left=None, right=None, root=None, val=None):
        self.left = left
        self.right = right
        self.root = root
        self.val = val

def build_tree(node, string, i):
    if node:
        if isinstance(node, str): # If the node value is a character then print it
            string += node
            return string
        else:  # Otherwise, traverse tree
            string = build_tree(node.left, string, '0')
            string = build_tree(node.right, string, '1')
    return string

def huffman_code_tree(string):
    frequency = defaultdict(int) # Dictionary with characters as keys and frequencies as values
    for char in string:
        if isinstance(char, str) and char not in string.punctuation + string.whitespace:
            frequency[char] += 1

    heap = [[weight, [node, ""]] for node, weight in frequency.items()] # Create a min-heap

    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)

        for pair in lo[1:]+hi[1:]:
            pair[1] = '0' + pair[1] if pair[1].startswith('0') else '1' + pair[1] # Create code

        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return sorted(build_tree(*heappop(heap)[1:]), key=len)[-1][::-1] # Return the last character's code