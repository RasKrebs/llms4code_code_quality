import psutil
import os
import threading
import multiprocessing
from typing import List

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



class RabinKarp:
    def __init__(self, text: str, pattern: str, base: int = 256, prime: int = 101):
        self.text = text
        self.pattern = pattern
        self.base = base
        self.prime = prime

    def search(self) -> List[int]:
        """
        Search for the pattern in the text using Rabin-Karp algorithm.

        Returns:
        List of indices where the pattern is found in the text.
        """
        results = []
        m = len(self.pattern)
        n = len(self.text)
        pattern_hash = self.hash(self.pattern, m)
        text_hash = self.hash(self.text[:m], m)

        for i in range(n - m + 1):
            if pattern_hash == text_hash and self.text[i:i+m] == self.pattern:
                results.append(i)
            if i < n - m:
                text_hash = self.recalculate_hash(text_hash, self.text[i], self.text[i+m], m)
        
        return results

    def hash(self, s: str, length: int) -> int:
        """
        Calculate the hash value of a string.

        Args:
        s: The string to calculate the hash for.
        length: Length of the string.

        Returns:
        Hash value of the string.
        """
        hash_value = 0
        for char in s:
            hash_value = (hash_value * self.base + ord(char)) % self.prime
        return hash_value

    def recalculate_hash(self, old_hash: int, old_char: str, new_char: str, length: int) -> int:
        """
        Recalculate the hash value based on the old hash, old character, new character and length.

        Args:
        old_hash: Hash value of the old string.
        old_char: Character being removed.
        new_char: Character being added.
        length: Length of the string.

        Returns:
        Recalculated hash value.
        """
        old_hash = (old_hash - ord(old_char) * (self.base**(length-1))) % self.prime
        new_hash = (old_hash * self.base + ord(new_char)) % self.prime
        return new_hash
def execute():

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
    
    # Initialize the RabinKarp object
    rk = RabinKarp(text, pattern)
    
    # Utilize the RabinKarp object's search method to find the pattern in the text
    position = rk.search()



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

