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
