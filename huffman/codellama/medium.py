def generate_huffman_node_tree(string):
    """Generate a Huffman node tree from a given string.

    Args:
        string (str): Input string for which to create the tree.

    Returns:
        root_node (Node): Root node of the generated Huffman tree.
    """

    # Generate frequency dictionary mapping characters to their frequencies
    freq = {}
    for char in string:
      if char not in freq:
          freq[char] = 1
      else:
          freq[char] += 1

    nodes = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    while len(nodes) > 1:
        left, f1 = nodes[-1]
        right, f2 = nodes[-2]
        nodes = nodes[:-2]

        # Create internal node with character '_' and frequency sum of left and right children
        new_node = Node('_', f1 + f2)
        new_node.left_child = left
        new_node.right_child = right

        if not nodes:
            nodes.append((new_node, f1 + f2))

        # Insert the new node in the sorted list while maintaining sorting order according to frequency
        else:
          for i, (node, freq) in enumerate(nodes):
              if freq <= f1 + f2:
                  nodes.insert(i, (new_node, f1 + f2))
                  break

    root_node = nodes[0][0]
    return root_node

def huffman_code_tree(root_node):
    """Generate a dictionary mapping characters to their Huffman codes.

    Args:
        root_node (Node): Root node of the Huffman tree.

    Returns:
        code_dict (dict): Dictionary mapping characters to their Huffman codes.
    """

    code_dict = {}
    # Traverse the tree recursively and generate huffman code for each character
    def traverse(node, current_code=""):
      if node is None:
        return

      if node.character != '_':
          code_dict[node.character] = current_code

      # Recursively traverse the tree and add 0 to huffman codes for left children and 1 for right children
      traverse(node.left_child, current_code + "0")
      traverse(node.right_child, current_code + "1")

    traverse(root_node)
    return code_dict

class Node:
    """A class representing a node in the Huffman tree."""

    def __init__(self, character, frequency):
        self.character = character  # Character represented by this node (or '_' for internal nodes)
        self.frequency = frequency  # Frequency of the character or subtree represented by this node
        self.left_child = None     # Left child node
        self.right_child = None    # Right child node

# Compression Function:
def compress(input_string):
  """Function to compress a given string using Huffman coding algorithm."""

  # Generate the huffman tree and get the code dictionary
  root_node = generate_huffman_node_tree(input_string)
  code_dict = huffman_code_tree(root_node)

  compressed_output = ""
  for char in input_string:
    compressed_output += code_dict[char]

  return compressed_output