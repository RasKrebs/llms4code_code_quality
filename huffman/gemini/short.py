import heapq

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text):
    """Builds a Huffman tree from the frequency of characters in the text."""
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1

    heap = []
    for char, f in freq.items():
        heapq.heappush(heap, Node(char, f))

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        new_node = Node(None, node1.freq + node2.freq)
        new_node.left = node1
        new_node.right = node2
        heapq.heappush(heap, new_node)

    return heapq.heappop(heap)

def generate_codes(root, code=''):
    """Generates Huffman codes for each character in the tree."""
    codes = {}
    if root is None:
        return codes

    if root.char is not None:
        codes[root.char] = code

    codes.update(generate_codes(root.left, code + '0'))
    codes.update(generate_codes(root.right, code + '1'))

    return codes

def encode(text, codes):
    """Encodes the text using Huffman codes."""
    encoded_text = ''
    for char in text:
        encoded_text += codes[char]
    return encoded_text

def compress(text):
    """Compresses the text using Huffman coding."""
    tree = build_huffman_tree(text)
    codes = generate_codes(tree)
    encoded_text = encode(text, codes)
    return encoded_text