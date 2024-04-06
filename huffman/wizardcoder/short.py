import heapq
from collections import defaultdict

class Node(object):
    def __init__(self, character=None, frequency=0, left=None, right=None):
        self.character = character
        self.frequency = frequency
        self.left = left
        self.right = right

    def is_leaf(self):
        return not (self.right or self.left)

    # Define __lt__ for heapq comparison
    def __lt__(self, other):
        return self.frequency < other.frequency

def huffman_coding(string):
    frequency = defaultdict(int)
    for char in string:
        frequency[char] += 1

    heap = [Node(character=char, frequency=freq) for char, freq in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)

        merged_node = Node(frequency=node1.frequency+node2.frequency, left=node1, right=node2)

        heapq.heappush(heap, merged_node)

    root = heap[0]

    # Create Huffman Code
    huffman_code = {}
    def create_huffman_code(root, current_bits):
        if root is None:
            return

        if root.is_leaf():
            huffman_code[root.character] = ''.join(current_bits)

        current_bits.append('0')
        create_huffman_code(root.left, current_bits)
        current_bits.pop()

        current_bits.append('1')
        create_huffman_code(root.right, current_bits)
        current_bits.pop()

    create_huffman_code(root, [])

    # Encode string
    encoded_string = ''.join([huffman_code[char] for char in string])

    return huffman_code, encoded_string
