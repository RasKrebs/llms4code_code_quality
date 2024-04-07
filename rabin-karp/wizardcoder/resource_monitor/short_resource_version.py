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
    p = len(pattern) # length of the pattern
    t = len(text)  # length of the text
    base = 26
    modulus = 1000003

    # Compute the hash value for pattern and first window of text.
    pattern_hash = 0
    text_hash = 0
    h = 1
    i = 0

    while i < p:
        pattern_hash = (base * pattern_hash + ord(pattern[i])) % modulus
        text_hash = (base * text_hash + ord(text[i])) % modulus
        if i != p-1:
            h = (h*base) % modulus  # Update hash value for rolling string.
        i += 1

    for s in range(t - p + 1):
        if pattern_hash == text_hash and text[s:s+p] == pattern:
            return s

        if s < t - p:  # Update hash value for next window of text.
            old = ord(text[s])
            new = ord(text[s + p])
            text_hash = (base * (text_hash - h * old) + new) % modulus

    return -1  # Pattern not found in the text.
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

