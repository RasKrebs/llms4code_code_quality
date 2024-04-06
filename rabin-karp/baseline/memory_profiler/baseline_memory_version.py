
# IMPORTED FROM EVALUATE SCRIPT
from memory_profiler import profile
import os
import psutil

# Get the current process ID
pid = os.getpid()

# Create a psutil Process object for the current process
process = psutil.Process(pid)

# Get the number of logical CPUs in the system
num_cores = psutil.cpu_count(logical=True)

# BELOW IMPORTS ARE FROM THE SOLUTION
        
# Following program is the implementation of Rabin Karp Algorithm for Pattern Searching
# The code is taken from https://www.geeksforgeeks.org/rabin-karp-algorithm-for-pattern-searching/
@profile
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
