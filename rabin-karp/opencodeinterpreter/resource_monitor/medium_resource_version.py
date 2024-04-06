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
    def __init__(self, text_string: str, pattern_string: str, base: int = 256, prime: int = 101):
        """
        Initializes the RabinKarp class with a given text string, pattern string, and optional base and prime numbers.

        Parameters:
            text_string (str): The string in which to find the pattern.
            pattern_string (str): The string that we are searching for within the text string.
            base (int): The base number used in the hash function. Default is 256.
            prime (int): A prime number larger than the length of the pattern string. It is used to compute the hash value and avoid collisions.
        """
        self.text_string = text_string
        self.pattern_string = pattern_string
        self.base = base
        self.prime = prime

    def search(self) -> list:
        """
        Searches the given text string for all occurrences of the pattern string using the Rabin-Karp algorithm.

        Returns:
            matches (list): A list containing all start indices in the text string where a match with the pattern string is found.

        """
        m = len(self.pattern_string)
        n = len(self.text_string)

        # Compute hash values for pattern and first window of text.
        pattern_hash = 0
        text_hash = 0
        h = 1
        matches = []

        for i in range(m-1):
            h = (h*self.base)%self.prime

        for i in range(m):
            pattern_hash = (self.base * pattern_hash + ord(self.pattern_string[i])) % self.prime
            text_hash = (self.base * text_hash + ord(self.text_string[i])) % self.prime

        # Slide the window over the text, one character at a time.
        for i in range(n-m+1):
            if pattern_hash == text_hash:
                j = 0
                while j < m:
                    if self.text_string[i+j] != self.pattern_string[j]:
                        break

                    j += 1

                # If we've matched all characters in the pattern string, add this match to our list of matches.
                if j == m:
                    matches.append(i)

            # Slide the window over one character and recompute the hash for the new window.
            if i < n-m:
                text_hash = (self.base*(text_hash - ord(self.text_string[i])*h) + ord(self.text_string[i+m])) % self.prime

                # Ensure that the hash is a positive number.
                if text_hash < 0:
                    text_hash += self.prime

        return matches
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

