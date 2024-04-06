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
    if len(pattern) > len(text):
        return -1

    q = 2**31-1  # large prime number
    d = 256   # number of unique characters in text/pattern
    h = pow(d, len(pattern)-1) % q  # precomputed value: (d^m-1) mod q

    t_hash = 0  # hash for the text substring
    p_hash = 0  # hash for the pattern

    # calculate initial hash values
    for i in range(len(pattern)):
        t_hash = (d*t_hash + ord(text[i])) % q
        p_hash = (d*p_hash + ord(pattern[i])) % q

    # slide pattern over text and check hash values for equality
    for s in range(len(text)-len(pattern)+1):
        if t_hash == p_hash:
            match = True
            for i in range(len(pattern)):
                if text[s+i] != pattern[i]:
                    match = False
                    break
            if match:
                return s  # found a match

        if s < len(text)-len(pattern):
            t_hash = ((d*(t_hash - h*ord(text[s])) + ord(text[s+len(pattern)])) % q)

    return -1  # no match found
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

