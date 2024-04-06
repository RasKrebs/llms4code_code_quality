from dataclasses import dataclass
from heapq import heappush, heappop

@dataclass
class Node:
    value: str
    freq: int
    left: 'Node' = None
    right: 'Node' = None

    def __lt__(self, other):
        return self.freq < other.freq

def generate_freq_table(data: str) -> dict:
    freq_table = {}
    for char in data:
        if char not in freq_table:
            freq_table[char] = 0
        freq_table[char] += 1
    return freq_table

def build_huffman_tree(freq_table: dict) -> Node:
    heap = []
    for char, freq in freq_table.items():
        heappush(heap, Node(char, freq))

    while len(heap) > 1:
        left = heappop(heap)
        right = heappop(heap)
        merged = Node(None, left.freq + right.freq, left, right)
        heappush(heap, merged)

    return heap[0]

def generate_codes(node: Node, current_code: str, codes: dict):
    if node is None:
        return

    if node.value is not None:
        codes[node.value] = current_code
        return

    generate_codes(node.left, current_code + '0', codes)
    generate_codes(node.right, current_code + '1', codes)

def huffman_encode(data: str) -> str:
    freq_table = generate_freq_table(data)
    tree = build_huffman_tree(freq_table)
    codes = {}
    
    generate_codes(tree, '', codes)
    
    encoded_data = ''.join(codes[char] for char in data)
    
    return encoded_data, tree

def huffman_decode(encoded_data: str, tree: Node) -> str:
    decoded_data = ''
    current_node = tree

    for bit in encoded_data:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right

        if current_node.value is not None:
            decoded_data += current_node.value
            current_node = tree

    return decoded_data
