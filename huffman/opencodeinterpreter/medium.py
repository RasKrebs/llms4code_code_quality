# A Huffman Node
class NodeTree(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def nodes(self):
        return (self.left, self.right)

# A Huffman Tree
class HuffmanTree(object):
    def __init__(self, root: NodeTree):
        self.root = root

    def traverse_tree(self, node: NodeTree, code=''):
        if type(node) is str:
            return {node: code}
        (l, r) = node.nodes()
        d = dict()
        d.update(self.traverse_tree(l, code + '0'))
        d.update(self.traverse_tree(r, code + '1'))
        return d

def calculate_frequency(text): # 
    """
    Calculate the frequency of each character in the given text.
    """
    frequency = {}
    for character in text:
        if character not in frequency:
            frequency[character] = 1
        else:
            frequency[character] += 1
    return frequency

def build_huffman_tree(frequencies):
    """
    Build the Huffman tree based on the frequencies of characters.
    """
    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    while len(heap) > 1:
        heap.sort()
        left = heap.pop(0)
        right = heap.pop(0)
        for pair in left[1:]:
            pair[1] = '0' + pair[1]
        for pair in right[1:]:
            pair[1] = '1' + pair[1]
        heap.append([left[0] + right[0]] + left[1:] + right[1:])
    return heap[0] if heap else None
