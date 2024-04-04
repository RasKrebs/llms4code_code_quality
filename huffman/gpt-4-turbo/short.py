import heapq
import collections

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # For comparing frequencies in priority queue
    def __lt__(self, other):
        return self.freq < other.freq

def build_frequency_dict(data):
    return collections.Counter(data)

def build_huffman_tree(freq_dict):
    priority_queue = [HuffmanNode(char, freq) for char, freq in freq_dict.items()]
    heapq.heapify(priority_queue)
    
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)

        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(priority_queue, merged)
    
    return priority_queue[0] if priority_queue else None

def encode_huffman_tree(root, binary_code="", binary_code_dict=None):
    if binary_code_dict is None:
        binary_code_dict = dict()

    if root:
        if root.char is not None:
            binary_code_dict[root.char] = binary_code
        encode_huffman_tree(root.left, binary_code + "0", binary_code_dict)
        encode_huffman_tree(root.right, binary_code + "1", binary_code_dict)

    return binary_code_dict

def compress_string(data):
    freq_dict = build_frequency_dict(data)
    root = build_huffman_tree(freq_dict)
    binary_code_dict = encode_huffman_tree(root)

    encoded_data = ''.join([binary_code_dict[char] for char in data])
    return encoded_data, binary_code_dict, root

def decode_data(encoded_data, binary_code_dict, root):
    decoded_data = ''
    current_node = root

    for bit in encoded_data:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right
        
        if current_node.char is not None:
            decoded_data += current_node.char
            current_node = root
    
    return decoded_data