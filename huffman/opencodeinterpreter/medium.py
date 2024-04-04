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