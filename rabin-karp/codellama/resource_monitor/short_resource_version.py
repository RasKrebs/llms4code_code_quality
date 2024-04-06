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



from typing import Union
import hashlib

# Define default hashing function (sha256)
DEFAULT_HASHING = hashlib.sha256().hexdigest()

def rabin_karp(text: str, pattern: str, hashing=DEFAULT_HASHING) -> Union[int, None]:
    """Returns the index of the first occurrence of the pattern in the text.
    Args:
        text (str): The string from which to find a substring.
        pattern (str): The pattern to search for within the given text.
        hashing (Union[str, callable], optional): Hashing algorithm used. Defaults to DEFAULT_HASHING.
    Returns:
        Union[int, None]: Index of first occurrence of the pattern in the text, or None if not found.
    """

    # Calculate hash value of pattern (assuming ASCII encoding)
    pattern_hash = hashing(pattern.encode("ascii"))

    # Calculate hash values for all substrings in text (assuming ASCII encoding)
    text_substrings = [text[i : i + len(pattern)] for i in range(len(text))]
    text_hashes = [hashing(s.encode("ascii")) for s in text_substrings]

    # Loop through all substrings in text
    for i, substring_hash in enumerate(text_hashes):
        # If the hash value matches, check if the full substring is equal to the pattern
        if substring_hash == pattern_hash and text[i : i + len(pattern)] == pattern:
            return i  # Return index of first occurrence of the pattern in the text
    # If no match has been found after searching all substrings, return None
    return None
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

