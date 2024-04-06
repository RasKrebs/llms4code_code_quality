from heapq import heappush, heappop, heapify
from collections import defaultdict

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_frequency_dict(data: str) -> dict:
    freq_dict = defaultdict(int)
    for char in data:
        freq_dict[char] += 1
    return freq_dict

def build_huffman_tree(freq_dict: dict) -> Node:
    priority_queue = []
    
    for char, freq in freq_dict.items():
        node = Node(char, freq)
        heappush(priority_queue, node)
        
    while len(priority_queue) > 1:
        node1 = heappop(priority_queue)
        node2 = heappop(priority_queue)
        
        merge_node = Node(None, node1.freq + node2.freq)
        merge_node.left = node1
        merge_node.right = node2
        
        heappush(priority_queue, merge_node)
        
    return priority_queue[0]

def build_huffman_codes(node, current_code="", codes={}):
    if node:
        if node.char:
            codes[node.char] = current_code
        build_huffman_codes(node.left, current_code + "0", codes)
        build_huffman_codes(node.right, current_code + "1", codes)
    
    return codes

def huffman_compress(data: str) -> str:
    if not isinstance(data, str):
        raise TypeError("Input data must be a string")
    
    freq_dict = build_frequency_dict(data)
    huffman_tree = build_huffman_tree(freq_dict)
    huffman_codes = build_huffman_codes(huffman_tree)
    
    compressed_data = "".join(huffman_codes[char] for char in data)
    
    return compressed_data
