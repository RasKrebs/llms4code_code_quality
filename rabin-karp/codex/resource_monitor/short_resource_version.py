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
    pattern_len = len(pattern)
    text_len = len(text)
    pattern_hash = 0
    text_hash = 0
    window_len = pattern_len
    d = 256  # number of characters in the input alphabet
    h = pow(d, pattern_len-1) % text_len  # precompute d^(m-1) % q

    # calculate the hash value of pattern and first window of text
    for i in range(0, pattern_len):
        pattern_hash = (d * pattern_hash + ord(pattern[i])) % text_len
        text_hash = (d * text_hash + ord(text[i])) % text_len

    # slide the pattern over text one by one
    for i in range(0, text_len-pattern_len+1):
        # check the hash values of current window of text and pattern
        # if the hash values match then only check for characters one by one
        if pattern_hash == text_hash:
            for j in range(0, pattern_len):
                if text[i+j] != pattern[j]:
                    break
            j += 1
            # if pattern_hash == text_hash and pattern[0...m-1] = text[i, i+1, ...i+m-1]
            if j == pattern_len:
                return i

        # calculate hash value for next window of text: remove leading digit, add trailing digit
        if i < text_len-pattern_len:
            text_hash = (d * (text_hash - ord(text[i])*h) + ord(text[i+pattern_len])) % text_len

            # we might get negative value of text_hash, converting it to positive
            if text_hash < 0:
                text_hash = text_hash + text_len

    return -1  # pattern not found in the text
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

