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
    # Prime number for modulus operation to reduce collisions
    prime = 101
    # Base for the polynomial rolling hash function
    d = 256
    
    # Lengths of text and pattern
    n = len(text)
    m = len(pattern)
    
    # Hash values for text and pattern
    hash_text = 0 
    hash_pattern = 0 
    
    # h = (d^(m-1)) % prime
    h = pow(d, m-1, prime)
    
    # Compute initial hash values for pattern and first window of text
    for i in range(m):
        hash_pattern = (d * hash_pattern + ord(pattern[i])) % prime
        hash_text = (d * hash_text + ord(text[i])) % prime
        
    # Slide the pattern over text one by one 
    for i in range(n - m + 1):
        # Check the hash values of current window of text and pattern
        # If the hash values match, then only check for characters one by one
        if hash_pattern == hash_text:
            # Check for characters one by one
            for j in range(m):
                if text[i+j] != pattern[j]:
                    break
            else:
                # Pattern found at index i
                return i
        
        # Calculate hash value for next window of text
        # Remove leading digit, add trailing digit
        if i < n-m:
            hash_text = (d*(hash_text - ord(text[i])*h) + ord(text[i+m])) % prime
            
            # We might get negative value of t, converting it to positive
            if hash_text < 0:
                hash_text = hash_text + prime
                
    # Pattern not found
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

