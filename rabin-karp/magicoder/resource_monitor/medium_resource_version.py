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
    def __init__(self, text: str, pattern: str, d: int = 256, q: int = 101):
        """
        Initialize the instance of class with given parameters.

        Parameters:
            text (str): The input string.
            pattern (str): The pattern to search in the text.
            d (int): A prime number for modulo operation. Defaults to 256.
            q (int): A base number to use in the hash function. Defaults to 101.
        """
        self.text = text
        self.pattern = pattern
        self.d = d
        self.q = q

    def _compute_hash(self, string: str) -> int:
        """
        Compute hash value of a given string.

        Parameters:
            string (str): The input string for which the hash needs to be computed.

        Returns:
            int: The hash value of the string.
        """
        length = len(string)
        h = 0
        for i in range(length):
            h = (h + ord(string[i]) * self.d ** (length - i - 1)) % self.q
        return h

    def _search(self) -> list:
        """
        Search the pattern in text using Rabin-Karp algorithm.

        Returns:
            list[int]: List of indices where pattern is found. Empty if not found.
        """
        M = len(self.pattern)
        N = len(self.text)
        p = 0    # hash value for pattern
        t = 0    # hash value for text
        h = 1
        result = []

        for _ in range(M-1):
            h = (h * self.d) % self.q

        # Calculate the hash of pattern and first window of text
        p = self._compute_hash(self.pattern)
        t = self._compute_hash(self.text[0:M])

        for i in range(N-M+1):
            if p == t:  # Check the hash values
                for j in range(M):
                    if self.text[i + j] != self.pattern[j]:    # character comparison
                        break

                j += 1
                if j == M:
                    result.append(i)  # Add the index to result list

            if i < N-M:
                t = (self.d * (t - ord(self.text[i]) * h) + ord(self.text[i + M])) % self.q

        return result

    def search(self):
        """
        This is the main function that calls _search() to find pattern in text using Rabin-Karp algorithm.

        Returns:
            list[int]: List of indices where pattern is found. Empty if not found.
        """
        return self._search()
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

