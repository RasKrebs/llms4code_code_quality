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

class RabinKarp:
    """
    A class to represent the Rabin-Karp algorithm.

    ...

    Attributes
    ----------
    base : int
        base for the hash function
    prime : int
        prime number for the hash function

    Methods
    -------
    search(text: str, pattern: str) -> int:
        Searches for the pattern in the text.
    """

    def __init__(self, base: Optional[int] = 256, prime: Optional[int] = 101):
        """
        Constructs all the necessary attributes for the RabinKarp object.

        Parameters
        ----------
        base : int, optional
            base for the hash function (default is 256)
        prime : int, optional
            prime number for the hash function (default is 101)
        """

        self.base = base
        self.prime = prime

    def search(self, text: str, pattern: str) -> int:
        """
        Searches for the pattern in the text.

        Parameters
        ----------
        text : str
            The text to search in
        pattern : str
            The pattern to search for

        Returns
        -------
        int
            The index of the first occurrence of the pattern in the text, or -1 if the pattern is not found
        """

        n = len(text)
        m = len(pattern)
        pattern_hash = 0
        text_hash = 0
        h = 1

        # Calculate h = (base^(m-1)) % prime
        for _ in range(m - 1):
            h = (h * self.base) % self.prime

        # Calculate hash value for pattern and text
        for i in range(m):
            pattern_hash = (self.base * pattern_hash + ord(pattern[i])) % self.prime
            text_hash = (self.base * text_hash + ord(text[i])) % self.prime

        # Slide the pattern over text one by one
        for i in range(n - m + 1):
            # Check the hash values of current window of text and pattern
            if pattern_hash == text_hash:
                # Check for characters one by one
                for j in range(m):
                    if text[i + j] != pattern[j]:
                        break
                j += 1
                # If pattern_hash == text_hash and pattern[0...m-1] = text[i, i+1, ...i+m-1]
                if j == m:
                    return i

            # Calculate hash value for next window of text: Remove leading digit, add trailing digit
            if i < n - m:
                text_hash = (self.base * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % self.prime

                # We might get negative values of text_hash, converting it to positive
                if text_hash < 0:
                    text_hash = text_hash + self.prime

        # If we reach here, then pattern is not present in text
        return -1
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

