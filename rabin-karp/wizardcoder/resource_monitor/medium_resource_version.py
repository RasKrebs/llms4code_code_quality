import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}


import numpy as np
np.random.seed(42)


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



from typing import List, Tuple

class RabinKarp:
    """A class implementing the Rabin-Karp string matching algorithm."""

    def __init__(self, base: int = 256, prime: int = 101):
        """Initialize a new instance of RabinKarp.

        :param base: The number of characters in the input alphabet (default is 256 for ASCII)
        :type base: int
        :param prime: A large prime number to mod our hashes with (default is 101)
        :type prime: int
        """
        self.base = base
        self.prime = prime

    def _hash(self, string: str, start_idx: int, length: int) -> Tuple[int, List[int]]:
        """Compute the hash of a substring and pre-compute hashes for all prefixes.

        :param string: The input string to compute hashes on
        :type string: str
        :param start_idx: Index at which to begin computing hashes (default is 0)
        :type start_idx: int
        :param length: Length of the substring for which to compute the hash
        :type length: int
        :return: A tuple containing the computed hash and pre-computed hashes
        :rtype: Tuple[int, List[int]]
        """
        hashes = [0] * (len(string) + 1)
        power = 1
        for i in range(start_idx + length - 1, start_idx - 1, -1):
            hashes[i] = ((hashes[i+1] * self.base) % self.prime + ord(string[i])) % self.prime
            power = (power * self.base) % self.prime
        return hashes[start_idx], hashes, power

    def find_pattern(self, text: str, pattern: str) -> List[int]:
        """Find the starting indices of all occurrences of the pattern in the text.

        :param text: The input string to search for the pattern
        :type text: str
        :param pattern: The pattern to look for in the text
        :type pattern: str
        :return: A list containing starting indices of all occurrences of the pattern in the text
        :rtype: List[int]
        """
        pat_hash, _, _ = self._hash(pattern, 0, len(pattern))
        _, txt_hashes, power = self._hash(text, 0, len(pattern))

        indices = []
        for i in range(len(text) - len(pattern) + 1):
            if pat_hash == txt_hashes[i]:
                if text[i : i + len(pattern)] == pattern:
                    indices.append(i)
            if i < len(text) - len(pattern):  # prevent out of bounds error
                txt_hashes[i] = ((txt_hashes[i+1] - power * ord(text[i])) % self.prime + self.prime) % self.prime
        return indices
def execute():
    # Initialize the RabinKarp object
    rk = RabinKarp()

    # Define a text to search within
    text = """
Integration of Artificial Intelligence (AI) into business processes has garnered significant 
attention, with particularly automated code generation emerging as a promising area. Tools
like ChatGPT and GitHub Copilot are increasingly being implemented in development pro-
cesses, hinting at potential gains in developer efficiency and productivity (Poldrack et al.,
2023). While efficiency and productivity is advantageous, it represents only one facet of
a successful development process. Despite this, there remains little research exploring the
value added and effects on the software development process of such tools. Furthermore,
integrating new technologies in any business process is often accompanied by establishing
frameworks that outline their effective use. Such methodologies are crucial for effectively
harnessing the potential of these innovations but have yet to be extensively explored in the
current context. MOTIVATION Tools like ChatGPT and GitHub Copilot are increas-
ingly being implemented in development processes, hinting at potential gains in developer
efficiency and productivity"""
    
    # Define a pattern to search for in the text
    pattern = "establishing frameworks"
    
    # Utilize the RabinKarp object's search method to find the pattern in the text
    position = rk.search(text, pattern)



if __name__ == "__main__":
    # Start the resource monitoring in a separate thread
    global monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=resource_monitor)
    monitor_thread.start()

    # Execute the Huffman coding process

    # Using the execute function
    output = execute()


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

