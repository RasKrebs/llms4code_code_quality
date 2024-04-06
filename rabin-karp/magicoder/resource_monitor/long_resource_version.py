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
    """Rabin-Karp algorithm class"""

    def __init__(self, pattern_str: str, text_str: str, base=256, prime=101):
        """Initialize the pattern and text strings along with default values for base and prime number.

        Parameters:
            pattern_str (str): The string pattern to be searched in a larger string
            text_str (str): The larger string where pattern is being searched
            base (int, optional): Base used by the hash function Defaults to 256.
            prime (int, optional): Prime number for mod operation in hash calculation Defaults to 101.
        """
        self.pattern_str = pattern_str
        self.text_str = text_str
        self.base = base
        self.prime = prime

    def rabin_karp_search(self)-> int:
        """Implementation of the Rabin-Karp algorithm for pattern searching in a larger string.

        Returns:
            int: The index at which the pattern first occurs, -1 if not found.
        """

        pattern_len = len(self.pattern_str)
        text_len = len(self.text_str)
        prime = self.prime

        # Compute the hash value of the pattern and first window of text
        pattern_hash = 0   # Pattern Hash
        text_hash = 0      # Text Hash
        h = 1

        for i in range(-1):
            h = (h * self.base) % prime

        # Calculate the hash values for pattern and first window of text
        for i in range(pattern_len):
            pattern_hash += ord(self.pattern_str[i]) * h
            text_hash += ord(self.text_str[i]) * h

        pattern_hash = pattern_hash % prime
        text_hash = text_hash % prime

        # Slide the pattern over text one by one
        for i in range(text_len - pattern_len + 1):
            if pattern_hash == text_hash:
                return i   # Pattern found at position i

            # Calculate next window hash value
            if i < text_len - pattern_len:
                text_hash = (self.base * (text_hash - ord(self.text_str[i])*h) + ord(self.text[i+pattern_len])) % prime

                if text_hash < 0:
                    text_hash += prime  # Avoid negative hash values

        return -1   # Pattern not found in text
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

