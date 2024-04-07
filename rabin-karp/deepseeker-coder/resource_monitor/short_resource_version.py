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
    d = 256  # number of characters in the input alphabet
    q = 101  # a prime number
    m = len(pattern)
    n = len(text)
    p = 0  # hash value for pattern
    t = 0  # hash value for text
    h = 1

    # The value of h would be "pow(d, M-1)%q"
    for i in range(m-1):
        h = (h*d)%q

    # Calculate the hash value of pattern and first window of text
    for i in range(m):
        p = (d*p + ord(pattern[i]))%q
        t = (d*t + ord(text[i]))%q

    # Slide the pattern over text one by one
    for s in range(n-m+1):
        # Check the hash values of current window of text and pattern
        # If the hash values match then only check for characters on by one
        if p == t:
            # Check for characters one by one
            for j in range(m):
                if text[s+j] != pattern[j]:
                    break
            j+=1
            # if p == t and pattern[0...M-1] = txt[i, i+1, ...i+M-1]
            if j == m:
                return s

        # Calculate hash value for next window of text: Remove leading digit, add trailing digit
        if s < n-m:
            t = (d*(t-ord(text[s])*h) + ord(text[s+m]))%q

            # We might get negative values of t, converting it to positive
            if t < 0:
                t = t+q
    return -1
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

