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
    """
    A class to represent the Rabin-Karp algorithm for string pattern matching.

    ...

    Attributes
    ----------
    text : str
        the text string to search in
    pattern : str
        the pattern string to search for
    base : int
        the base to use in the hash function
    prime : int
        the prime number to use in the hash function

    Methods
    -------
    search():
        Searches for the pattern in the text and returns the starting indices of all occurrences.
    """

    def __init__(self, text: str, pattern: str, base: int = 256, prime: int = 101):
        """
        Constructs all the necessary attributes for the RabinKarp object.

        Parameters
        ----------
            text : str
                the text string to search in
            pattern : str
                the pattern string to search for
            base : int, optional
                the base to use in the hash function (default is 256)
            prime : int, optional
                the prime number to use in the hash function (default is 101)
        """

        self.text = text
        self.pattern = pattern
        self.base = base
        self.prime = prime

    def search(self) -> List[int]:
        """
        Searches for the pattern in the text and returns the starting indices of all occurrences.

        Returns
        -------
        list
            a list of starting indices of all occurrences of the pattern in the text
        """

        n = len(self.text)
        m = len(self.pattern)
        pattern_hash = self._hash(self.pattern, m)
        text_hash = self._hash(self.text, m)

        occurrences = []

        for i in range(n - m + 1):
            if pattern_hash == text_hash:
                if self.pattern == self.text[i:i+m]:
                    occurrences.append(i)

            if i < n - m:
                text_hash = self._rehash(self.text, i, i+m, text_hash, m)

        return occurrences

    def _hash(self, string: str, end: int) -> int:
        """
        Calculates the hash value of a substring of the given string.

        Parameters
        ----------
        string : str
            the string to calculate the hash of
        end : int
            the end index of the substring

        Returns
        -------
        int
            the hash value of the substring
        """

        hash_value = 0
        for i in range(end):
            hash_value = (hash_value * self.base + ord(string[i])) % self.prime
        return hash_value

    def _rehash(self, string: str, old_index: int, new_index: int, old_hash: int, length: int) -> int:
        """
        Re-calculates the hash value of a substring of the given string based on the old hash value.

        Parameters
        ----------
        string : str
            the string to calculate the hash of
        old_index : int
            the old start index of the substring
        new_index : int
            the new start index of the substring
        old_hash : int
            the old hash value of the substring
        length : int
            the length of the substring

        Returns
        -------
        int
            the new hash value of the substring
        """

        new_hash = (old_hash - ord(string[old_index]) * pow(self.base, length - 1, self.prime)) % self.prime
        new_hash = (new_hash * self.base + ord(string[new_index])) % self.prime
        return new_hash
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

