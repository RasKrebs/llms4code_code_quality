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
