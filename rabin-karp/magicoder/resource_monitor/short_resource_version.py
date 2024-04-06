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



def rabinkarp(text, pattern):
    len_t = len(text)
    len_p = len(pattern)

    # Check if length of text is less than pattern, return -1.
    if len_t < len_p:
        return -1

    prime = 101  # Prime number to avoid collisions
    hash_text = [0]*len(text)
    hash_pattern = [0]*len(pattern)

    # Calculate the hash for text and pattern.
    for i in range(len_p):
        hash_text[i] = (ord(text[i]) - ord('a') + 1) % prime  # ASCII value of 'a' is assumed to be 97.
        hash_pattern[i] = (ord(pattern[i]) - ord('a') + 1) % prime

    hash_t = 0;

    for i in range(len_p):
        hash_t = (hash_t + hash_text[i]) % prime

    # Calculate the rolling hash
    h = 1
    for i in range(len_p-1):  # Use only len_p-1 because we have already calculated hash value of first 'n' characters.
        h = (h*26)%prime  # 26 is the number of lower case alphabets, can be changed if there are other kinds of characters in text or pattern.

    rolling_hash = 0

    for i in range(len_t - len_p + 1):   # Iterating from (i to n-m+1)
        if rolling_hash == hash_t:  # If both hashes are equal, check if text and pattern matches.
            if text[i : i + len_p] == pattern:  # If matched return index.
                return i

        # Calculate the next window's hash value by subtracting first character of previous window and adding last character of new window.
        if i < len_t - len_p:
            rolling_hash = (26 * (rolling_hash - hash_text[i]*h) + hash_text[i+len_p])%prime  # Modulus is done to keep the value in the range of prime.

    return -1  # Return -1 if pattern not found.
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

