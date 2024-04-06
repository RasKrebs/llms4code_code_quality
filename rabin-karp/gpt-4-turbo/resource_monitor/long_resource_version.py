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
    Rabin-Karp Algorithm implementation for pattern matching.
    This class implements the Rabin-Karp Algorithm, which utilizes a rolling
    hash function for efficient substring search. The algorithm matches the
    hash value of the pattern with substring hash values of the text.
    """

    def __init__(self, base: int = 256, prime: int = 101):
        """
        Initialize the Rabin-Karp class with optional base and prime numbers.

        :param base: The base value for the rolling hash function. Default is 256.
        :param prime: The prime number to modulate the hash. Default is 101.
        """
        self.base = base
        self.prime = prime

    def search(self, text: str, pattern: str) -> int:
        """
        Search for the pattern in the given text using the Rabin-Karp algorithm.

        :param text: The text in which to search for the pattern.
        :param pattern: The pattern to search for.
        :return: The index of the first occurrence of the pattern in the text, or -1 if not found.
        """
        n = len(text)
        m = len(pattern)
        if m > n:
            return -1
        
        pattern_hash = 0
        text_hash = 0
        h = 1  # The value for the highest digit's contribution to hash

        # Pre-processing: Calculate the hash value of the pattern and the first window of the text
        for i in range(m):
            if i > 0:
                h = (h * self.base) % self.prime
            pattern_hash = (self.base * pattern_hash + ord(pattern[i])) % self.prime
            text_hash = (self.base * text_hash + ord(text[i])) % self.prime

        # Slide the pattern over text one by one and check for matches
        for i in range(n - m + 1):
            if pattern_hash == text_hash:
                # Check characters one by one for a sure match
                for j in range(m):
                    if text[i + j] != pattern[j]:
                        break
                else:  # Match found
                    return i

            if i < n - m:
                # Calculate hash value for the next window: Remove leading digit, add trailing digit
                text_hash = (self.base * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % self.prime

                # We might get a negative value of text_hash, converting it to positive
                if text_hash < 0:
                    text_hash = text_hash + self.prime

        return -1  # Pattern not found
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

