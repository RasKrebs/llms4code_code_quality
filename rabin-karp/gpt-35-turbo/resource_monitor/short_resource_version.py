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
    if not text or not pattern:
        return -1

    def hash_func(s):
        h = 0
        for char in s:
            h = (h * 256 + ord(char)) % q
        return h

    def rehash(prev_hash, prev_char, next_char, length):
        new_hash = (prev_hash - ord(prev_char) * pow(256, length - 1, q)) * 256 + ord(next_char)
        return new_hash % q

    n = len(text)
    m = len(pattern)
    q = 997
    h_pattern = hash_func(pattern)
    h_text = hash_func(text[:m])

    if h_text == h_pattern and text[:m] == pattern:
        return 0

    for i in range(1, n - m + 1):
        h_text = rehash(h_text, text[i - 1], text[i + m - 1], m)
        if h_text == h_pattern and text[i:i+m] == pattern:
            return i

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

