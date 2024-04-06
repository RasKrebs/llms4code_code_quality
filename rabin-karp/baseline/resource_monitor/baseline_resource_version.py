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



# Following program is the implementation of Rabin Karp Algorithm for Pattern Searching
# The code is taken from https://www.geeksforgeeks.org/rabin-karp-algorithm-for-pattern-searching/
def rabin_karp(pattern:str,
               text:str,
               prime:int = 101,
               base:int = 256) -> int:

    """Rabin Karp Algorithm for Pattern Searching

    Args:
        patern (str): Pattern to be searched
        text (str): Text in which pattern is to be searched
        prime (int, optional): Prime number used for hashing. Defaults to 101.
        base (int, optional): Number of characters in the input alphabet. Defaults to
            256 which is equal to ASCI.

    Returns:
        Integer equal to the index of the first occurrence of the pattern in the text
    """
    # Extra pattern and text length
    pattern_len = len(pattern)
    text_len = len(text)

    # hash value for pattern
    pattern_hash = 0

    # hash value for text
    text_hash = 0
    hash = 1

    # The value of h would be "pow(d, pattern_len-1)%q"
    for i in range(pattern_len-1):
        hash = (hash * base) % prime

    # Calculate the hash value of pattern and first window of text
    for i in range(pattern_len):
        pattern_hash = (base * pattern_hash + ord(pattern[i])) % prime
        text_hash = (base * text_hash + ord(text[i])) % prime

    # Slide the pattern over text one by one
    for i in range(text_len-pattern_len+1):
        # Check the hash values of current window of text and
        # pattern if the hash values pattern_lenatch then only check
        # for characters one by one
        if pattern_hash == text_hash:
            # Check for characters one by one
            for j in range(pattern_len):
                if text[i+j] != pattern[j]:
                    break
                j += 1

            # if p == t and pat[0...pattern_len-1] = txt[i, i+1, ...i+pattern_len-1]
            if j == pattern_len:
                print("Pattern found at index " + str(i))

        # Calculate hash value for next window of text: Remove leading digit, add trailing digit
        if i < text_len-pattern_len:
            text_hash = (base*(text_hash-ord(text [i])*hash) + ord(text[i+pattern_len])) % prime

            # We might get negative values of t, converting it to positive
            if text_hash < 0:
                text_hash = text_hash + prime
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

