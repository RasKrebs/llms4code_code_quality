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



def rabin_karp(text, pattern):
    """
    The Rabin-Karp algorithm for string pattern matching.

    Args:
        text (str): the text to search for the pattern in.
        pattern (str): the pattern to find in the text.

    Returns:
        List[int]: the starting positions of all occurrences of the pattern in the text.
        An empty list is returned if the pattern does not occur anywhere in the text.
    """

    # Preprocessing for Rabin-Karp algorithm
    prime = 101  # A prime number (small enough for simplicity)
    base = 256  # The alphabet size
    d = pow(base, len(pattern) - 1, prime)  # The base raised to the power of pattern length minus one modulo the prime.

    def rolling_hash(string):
        """
        Calculate the hash value for a string using Horner's method.

        Args:
            string (str): the input string to compute the hash for.

        Returns:
            int: the computed hash value for the given string.
        """
        h = 0
        for char in string:
            h = (h * base + ord(char)) % prime

        return h  # Compute the hash value of a pattern using Horner's method.

    pattern_hash = rolling_hash(pattern)  # Calculate the hash value for the pattern.

    def find_matches():
        """
        Find all occurrences of the pattern in the text and return their starting positions.

        Returns:
            List[int]: the starting positions of all occurrences of the pattern in the text.
        """
        result = []
        text_hash = rolling_hash(text[:len(pattern)])  # Calculate the hash value for the first len(pattern) characters in text.

        for i in range(len(text) - len(pattern) + 1):
            if pattern_hash != text_hash:
                if i < len(text) - len(pattern):
                    # Calculate the rolling hash using the pre-computed value and Horner's method.
                    text_hash = (base * (text_hash - ord(text[i]) * d) + ord(text[i + len(pattern)])) % prime
                else:  # No more characters left to hash.
                    break
            elif pattern == text[i:i + len(pattern)]:
                result.append(i)  # Append the starting position if we have a match.

        return result

    return find_matches()  # Return the list of starting positions for all occurrences of the pattern in the text.
def execute():
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
    rk = rabin_karp(text, pattern)



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

