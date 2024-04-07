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



class RabinKarpAlgorithm:
    def __init__(self, text: str, pattern: str, base: int = 256, prime: int = 101):
        self.text = text
        self.pattern = pattern
        self.base = base
        self.prime = prime
        self.text_len = len(text)
        self.pattern_len = len(pattern)
        self.base_power = pow(base, self.pattern_len - 1, prime)
        self.text_hash = self.hash_text(text)
        self.pattern_hash = self.hash_pattern(pattern)

    def hash_text(self, text: str) -> int:
        hash_value = 0
        for char in text:
            hash_value = (hash_value * self.base + ord(char)) % self.prime
        return hash_value

    def hash_pattern(self, pattern: str) -> int:
        hash_value = 0
        for char in pattern:
            hash_value = (hash_value * self.base + ord(char)) % self.prime
        return hash_value

    def check_equal_strings(self, text: str, start: int) -> bool:
        return text == self.text[start:start + self.pattern_len]

    def search(self) -> int:
        if self.pattern_len > self.text_len:
            return -1

        if self.pattern_hash == self.text_hash and self.check_equal_strings(self.text, 0):
            return 0

        for i in range(1, self.text_len - self.pattern_len + 1):
            # Recalculate hash value using rolling hash
            self.text_hash = (self.base * (self.text_hash - ord(self.text[i - 1]) * self.base_power) + ord(self.text[i + self.pattern_len - 1])) % self.prime

            # Check if hash values are equal and then verify with actual strings
            if self.text_hash == self.pattern_hash and self.check_equal_strings(self.text, i):
                return i

        return -1
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
    rk = RabinKarpAlgorithm(text, pattern)
    
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

