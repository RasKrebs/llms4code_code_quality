import heapq
from collections import defaultdict

class Node:
    def __init__(self, value, frequency):
        self.value = value
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency

def build_huffman_tree(text):
    frequency_map = defaultdict(int)
    for char in text:
        frequency_map[char] += 1

    pq = [Node(key, freq) for key, freq in frequency_map.items()]
    heapq.heapify(pq)

    while len(pq) > 1:
        left = heapq.heappop(pq)
        right = heapq.heappop(pq)

        merged = Node(None, left.frequency + right.frequency)
        merged.left = left
        merged.right = right
        heapq.heappush(pq, merged)

    return pq[0]

def build_code_mapping(root):
    code_mapping = {}

    def generate_code(node, code):
        if node:
            if node.value:
                code_mapping[node.value] = code
            generate_code(node.left, code + "0")
            generate_code(node.right, code + "1")

    generate_code(root, "")
    return code_mapping

def huffman_compress(text):
    root = build_huffman_tree(text)
    code_mapping = build_code_mapping(root)

    compressed_text = ""
    for char in text:
        compressed_text += code_mapping[char]

    return compressed_text
