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



from typing import Optional
class RabinKarpSearch:
    """
    Class implementing the Rabin-Karp algorithm for substring search in a given string.
    This class provides methods to set base and modulus values, and perform search operations.
    """
    def __init__(self, text: str, pattern: str):
        self.text = text
        self.pattern = pattern
        # Default values for base and modulus (prime numbers)
        self.base = 256
        self.modulus = 101

    def set_values(self, base: Optional[int]=None, modulus: Optional[int]=None):
        """
        Method to set base and modulus values (prime numbers) for the hash function.
        If not provided, default values will be used.
        """
        if base is not None:
            self.base = base
        if modulus is not None:
            self.modulus = modulus

    def search(self):
        """
        Method to find the first occurrence of the pattern in the text using Rabin-Karp algorithm.
        Returns the index of the start of the match, or -1 if no match is found.
        """
        # Calculate length of strings and check for edge cases
        N, M = len(self.text), len(self.pattern)
        if M > N: return -1

        # Pre-compute R^(M-1) % Q for use in removing leading digit
        RM = 1
        for _ in range(1, M):
            RM = (RM * self.base) % self.modulus

        pattern_hash = 0
        text_hash = 0

        # Compute hash for pattern and text's first window
        for i in range(M):
            pattern_hash = (self.base * pattern_hash + ord(self.pattern[i])) % self.modulus
            text_hash = (self.base * text_hash + ord(self.text[i])) % self.modulus

        # Slide the pattern over text one by one
        for i in range(N-M+1):
            # Check current window of text and pattern's hash values for match
            if text_hash == pattern_hash:
                # If hashes match, check characters one by one to confirm actual match
                j = 0
                while (j < M) :
                    if self.text[i+j] != self.pattern[j]: break
                    j += 1

                if j == M: return i # Found a match

            # Compute hash for next window of text, removing leading digit and adding trailing digit
            if i < N-M :
                text_hash = ((text_hash - RM*ord(self.text[i])) * self.base + ord(self.text[i+M])) % self.modulus

                # If hash is negative, make it positive
                if text_hash < 0 :
                    text_hash += self.modulus

        return -1 # No match found
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
    
    # Utilize the RabinKarp object's search method to find the pattern in the text
    rk = RabinKarpSearch(text, pattern)
    o = rk.search



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

