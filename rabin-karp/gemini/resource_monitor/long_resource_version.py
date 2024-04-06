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
   Implements the Rabin-Karp algorithm for pattern searching in a text string.
   """

   def __init__(self, base: int = 256, prime: int = 101):
       """
       Initializes the Rabin-Karp object with optional base and prime values for the hash function.

       Args:
           base: The base to use for the rolling hash function (default: 256).
           prime: The prime number for modulo operation in the hash function (default: 101).
       """

       self.base = base
       self.prime = prime

   def _hash(self, string: str) -> int:
       """
       Calculates the hash value of a string using the rolling hash function.

       Args:
           string: The string to hash.

       Returns:
           The hash value of the string.
       """

       hash_value = 0
       for char in string:
           hash_value = (hash_value * self.base + ord(char)) % self.prime
       return hash_value

   def search(self, text: str, pattern: str) -> int:
       """
       Searches for the first occurrence of a pattern in a text string using the Rabin-Karp algorithm.

       Args:
           text: The text string to search in.
           pattern: The pattern string to search for.

       Returns:
           The index of the first occurrence of the pattern in the text, or -1 if the pattern is not found.
       """

       n = len(text)
       m = len(pattern)

       # Handle edge case: pattern longer than text
       if m > n:
           return -1

       pattern_hash = self._hash(pattern)
       text_hash = self._hash(text[:m])  # Hash of first m characters of text

       for i in range(n - m + 1):
           if pattern_hash == text_hash:
               # Potential match, check for character-by-character equality
               if text[i : i + m] == pattern:
                   return i

           # Calculate hash for the next window in text
           if i < n - m:
               text_hash = (
                   (text_hash - ord(text[i]) * self.base**(m - 1)) * self.base + ord(text[i + m])
               ) % self.prime

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

