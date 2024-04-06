import heapq

class Node:
    def __init__(self, char, freq, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def calc_freq(s):
    freq = {}
    for char in s:
        if char not in freq:
            freq[char] = 0
        freq[char] += 1
    return freq

def build_huffman_tree(freq):
    heap = [Node(char, freq) for char, freq in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        heapq.heappush(heap, Node(None, left.freq + right.freq, left, right))
    return heap[0]  # root of the Huffman tree

def build_huffman_dict(node, binary_str='', huffman_dict={}):
    if node is None:
        return
    if node.char is not None:
        huffman_dict[node.char] = binary_str
    build_huffman_dict(node.left, binary_str + '0', huffman_dict)
    build_huffman_dict(node.right, binary_str + '1', huffman_dict)
    return huffman_dict

def huffman_encode(s):
    freq = calc_freq(s)
    root = build_huffman_tree(freq)
    huffman_dict = build_huffman_dict(root)
    return ''.join(huffman_dict[char] for char in s)
