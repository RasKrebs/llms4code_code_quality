def rabin_karp(text, pattern, d=256):
    """
    Rabin-Karp algorithm for pattern matching in text.

    Args:
        text: The text to search.
        pattern: The pattern to search for.
        d: The number of characters in the input alphabet (default 256).

    Returns:
        The index of the first occurrence of the pattern in the text,
        or -1 if the pattern is not found.
    """
    q = 101  # A prime number

    n = len(text)
    m = len(pattern)

    h = pow(d, m - 1) % q
    p = 0  # hash value for pattern
    t = 0  # hash value for text

    # Calculate hash values for pattern and first window of text
    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q

    # Slide the pattern over the text
    for i in range(n - m + 1):
        # Check if the hash values match
        if p == t:
            # If hash values match, check characters one by one
            for j in range(m):
                if text[i + j] != pattern[j]:
                    break
                else:
                    return i  # Pattern found

        # Calculate hash value for the next window of text
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % q

            return -1  # Pattern not found

    # Example usage
    text = "GEEKS FOR GEEKS"
    pattern = "FOR"

    index = rabin_karp(text, pattern)

    if index == -1:
        print("Pattern not found in the text")
    else:
        print("Pattern found at index", index)
