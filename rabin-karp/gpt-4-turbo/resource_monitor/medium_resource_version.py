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
    Rabin-Karp Algorithm class for string pattern matching.

    Attributes:
        text (str): The text string to be searched.
        pattern (str): The pattern string to be found in the text.
        base (int): Base value for hashing function. Default is 256 (number of ASCII characters).
        prime (int): A prime number to ensure a more uniform distribution of hash values. 
    """

    def __init__(self, text: str, pattern: str, base: int = 256, prime: int = 101):
        """
        Initializes the Rabin-Karp algorithm with the text and pattern strings,
        along with custom values for the base and prime numbers used in the hash function.
        """
        self.text = text
        self.pattern = pattern
        self.base = base
        self.prime = prime

    def search(self) -> list[int]:
        """
        Searches for all occurrences of the pattern in the text.

        Returns:
            A list of starting indices of all occurrences of the pattern in the text.
        """
        result = []  # To store the starting indexes of matched patterns
        M, N = len(self.pattern), len(self.text)
        i, j, p, t = 0, 0, 0, 0  # Initialize variables for pattern, text hash values and indexes
        h = 1  # The value of h would be "pow(d, M-1)%q"

        # Calculate the h value (pow(d, M-1) % q)
        for i in range(M-1):
            h = (h * self.base) % self.prime

        # Calculate the hash value of pattern and first window of text
        for i in range(M):
            p = (self.base * p + ord(self.pattern[i])) % self.prime
            t = (self.base * t + ord(self.text[i])) % self.prime

        # Slide the pattern over text one by one
        for i in range(N - M + 1):
            # Check the hash values of current window of text and pattern.
            # If the hash values match then only check for characters one by one
            if p == t:
                # Check for characters one by one
                for j in range(M):
                    if self.text[i + j] != self.pattern[j]:
                        break
                j += 1
                # if p == t and pattern[0...M-1] = text[i, i+1, ...i+M-1]
                if j == M:
                    result.append(i)

            # Calculate hash value for next window of text: Remove leading digit,
            # add trailing digit
            if i < N - M:
                t = (self.base * (t - ord(self.text[i]) * h) + ord(self.text[i + M])) % self.prime

                # We might get negative value of t, converting it to positive
                if t < 0:
                    t = t + self.prime

        return result
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
    
    # Initialize the RabinKarp object
    rk = RabinKarp(text, pattern)
    
    # Utilize the RabinKarp object's search method to find the pattern in the text
    position = rk.search()



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

