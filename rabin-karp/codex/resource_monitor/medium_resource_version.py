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



class RabinKarp:
    """
    Rabin-Karp algorithm for string pattern matching.
    """

    def __init__(self, base: int = 256, prime: int = 101):
        """
        Initialize the base and prime values for hash calculation.
        """
        self.base = base
        self.prime = prime

    def calculate_hash(self, text: str, end: int) -> int:
        """
        Calculate the hash value for a string.
        """
        hash_val = 0
        for i in range(end):
            hash_val = (self.base * hash_val + ord(text[i])) % self.prime
        return hash_val

    def recalculate_hash(self, text: str, old_index: int, new_index: int, old_hash: int, length: int) -> int:
        """
        Recalculate the hash value when sliding the window.
        """
        new_hash = ((old_hash - ord(text[old_index]) * pow(self.base, length - 1)) * self.base + ord(text[new_index])) % self.prime
        return new_hash

    def pattern_search(self, text: str, pattern: str) -> list:
        """
        Search for a pattern in a text string.
        """
        text_len = len(text)
        pattern_len = len(pattern)
        pattern_hash = self.calculate_hash(pattern, pattern_len)
        text_hash = self.calculate_hash(text, pattern_len)
        found_indexes = []

        for i in range(text_len - pattern_len + 1):
            if pattern_hash == text_hash and text[i:i+pattern_len] == pattern:
                found_indexes.append(i)
            if i < text_len - pattern_len:
                text_hash = self.recalculate_hash(text, i, i + pattern_len, text_hash, pattern_len)
        return found_indexes
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

