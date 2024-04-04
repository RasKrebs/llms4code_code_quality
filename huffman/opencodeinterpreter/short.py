class Node:
    def __init__(self, char=None, freq=None, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

def calculate_frequency(string):
    frequency = {}
    for char in string:
        if char not in frequency:
            frequency[char] = 1
        else:
            frequency[char] += 1
    return frequency

def build_heap(freq):
    heap = []
    for key in freq:
        node = Node(key, freq[key])
        heap.append(node)
    heap.sort(key=lambda x:x.freq)
    return heap

def build_huffman_tree(heap):
    while len(heap) > 1:
        node1 = heap.pop(0)
        node2 = heap.pop(0)
        merged = Node(left=node1, right=node2, freq=node1.freq + node2.freq)
        heap.append(merged)
        heap.sort(key=lambda x:x.freq)
    return heap[0]

def create_huffman_codes(root):
    current_code = ""
    codes = {}

    def traverse(node, code):
        if node is None:
            return
        if node.left is None and node.right is None:
            codes[node.char] = code
        traverse(node.left, code + "0")
        traverse(node.right, code + "1")
    traverse(root, current_code)
    return codes

def huffman_compress(string):
    frequency = calculate_frequency(string)
    heap = build_heap(frequency)
    root = build_huffman_tree(heap)
    codes = create_huffman_codes(root)
    compressed = ""
    for char in string:
        compressed += codes[char]
    return compressed