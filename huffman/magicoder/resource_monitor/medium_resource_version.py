import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}


import requests

book_url = "https://www.gutenberg.org/cache/epub/1513/pg1513-images.html"
response = requests.get(book_url)
text = response.text


# Implement the resource monitor
def resource_monitor():
    """
    Monitors the CPU and memory usage of the current process, updating global max usage.
    """
    global max_resources_usage
    process = psutil.Process(os.getpid())
    
    while monitoring:
        cpu_usage = process.cpu_percent(interval=1) / multiprocessing.cpu_count()
        memory_usage = process.memory_info().rss
        max_resources_usage['cpu'] = max(max_resources_usage['cpu'], cpu_usage)
        max_resources_usage['memory'] = max(max_resources_usage['memory'], memory_usage)



from typing import List, Dict
import heapq

class NodeTree(object):
    def __init__(self, char=None, left=None, right=None, frequency=0):
        self.char = char  
        self.frequency = frequency
        self.left = left  # Left child
        self.right = right  # Right child

    # Comparison methods for heapq
    def __lt__(self, other):
        return self.frequency < other.frequency


    # Added for resource monitoring
    def __eq__(self, other):
        return self.frequency == other.frequency

class HuffmanCoding():
    @staticmethod
    def calculate_frequency(s: str) -> Dict[str, int]:
        """
        Calculates frequency of each character in the string.

        Parameters:
            s (str): Input string for which frequencies are to be calculated.

        Returns:
            Dictionary with characters as keys and their corresponding frequencies as values.
        """
        freq = {}  # type: Dict[str, int]
        for c in s:
            if c not in freq:
                freq[c] = 0
            freq[c] += 1

        return freq

    @staticmethod
    def generate_heap(freq: Dict[str, int]) -> List[NodeTree]:
        """
        Generates a heap with frequency of each character.

        Parameters:
            freq (Dict): Dictionary containing characters and their frequencies in the string.

        Returns:
            A list representing min-heap that can be used to generate huffman tree.
        """
        heap = [[freq[key], key, NodeTree()] for key in freq]  # type: List[NodeTree]

        heapq.heapify(heap)
        return heap

    @staticmethod
    def build_tree(nodes: List[NodeTree]) -> NodeTree:
        """
        Generates the huffman tree based on the frequencies of each character.

        Parameters:
            nodes (List): A list representing min-heap that can be used to generate huffman tree.

        Returns:
            Root node of the generated Huffman Tree.
        """
        while len(nodes) > 1:
            lo = heapq.heappop(nodes)  # type: List[NodeTree]
            hi = heapq.heappop(nodes)  # type: List[NodeTree]

            left, right = lo[2], hi[2]
            parent_frequency = lo[0] + hi[0]
            parent_node = NodeTree(left=left, right=right)

            heapq.heappush(nodes, [parent_frequency, '', parent_node])

        return nodes[0][2]  # type: NodeTree

    @staticmethod
    def huffman_code_tree(root: NodeTree, s: str = '', mapping: Dict[str, str]=None) -> Dict[str, str]:
        """
        Generates the Huffman Codes for each character.

        Parameters:
            root (NodeTree): Root node of the generated huffman tree.

        Returns:
            Dictionary with characters as keys and their corresponding huffman codes as values.
        """
        if mapping is None:
            mapping = {}  # type: Dict[str, str]

        if root.left:
            HuffmanCoding.huffman_code_tree(root.left, s + '0', mapping)

        if root.right:
            HuffmanCoding.huffman_code_tree(root.right, s + '1', mapping)

        if not root.left and not root.right:
            mapping[root.char] = s  # type: ignore

        return mapping

    @staticmethod
    def compress(s: str) -> str:
        """
        Compresses the input string using huffman encoding.

        Parameters:
            s (str): Input string to be compressed.

        Returns:
            The Huffman encoded string.
        """
        freq = HuffmanCoding.calculate_frequency(s)  # type: Dict[str, int]

        heap = HuffmanCoding.generate_heap(freq)  # type: List[NodeTree]

        root = HuffmanCoding.build_tree(heap)  # type: NodeTree

        codes = HuffmanCoding.huffman_code_tree(root)  # type: Dict[str, str]

        return codes

    @staticmethod
    def decompress(s: str) -> str:
        """
        Decompresses the Huffman encoded input string.

        Parameters:
            s (str): Input string in huffman encoding format to be decompressed.

        Returns:
            The original string after being decompressed by the huffman algorithm.
        """
        freq = {}  # type: Dict[str, int]
        while len(s) % 8 != 0:
            s += '1'

        eof_index = -len(''.join([f"{ord(c):b}" for c in codes])) * 2

        codes = HuffmanCoding.huffman_code_tree(HuffmanCoding.build_tree(HuffmanCoding.generate_heap(freq)))  # type: Dict[str, str]

        reverse_mapping = {v: k for k, v in codes.items()}
        return ''.join([reverse_mapping[s[i*8: (i+1)*8]] for i in range(len(s)//8-eof_index//2)])  # type: ignore

def execute(text):
    root = HuffmanCoding.compress(text)  # Generate the Huffman tree from the input text
    return root



if __name__ == "__main__":
    # Start the resource monitoring in a separate thread
    global monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=resource_monitor)
    monitor_thread.start()

    # Execute the Huffman coding process
    huffman_codes = execute(text)


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

