def rabin_karp(text, pattern):
    p = len(pattern) # length of the pattern
    t = len(text)  # length of the text
    base = 26
    modulus = 1000003

    # Compute the hash value for pattern and first window of text.
    pattern_hash = 0
    text_hash = 0
    h = 1
    i = 0

    while i < p:
        pattern_hash = (base * pattern_hash + ord(pattern[i])) % modulus
        text_hash = (base * text_hash + ord(text[i])) % modulus
        if i != p-1:
            h = (h*base) % modulus  # Update hash value for rolling string.
        i += 1

    for s in range(t - p + 1):
        if pattern_hash == text_hash and text[s:s+p] == pattern:
            return s

        if s < t - p:  # Update hash value for next window of text.
            old = ord(text[s])
            new = ord(text[s + p])
            text_hash = (base * (text_hash - h * old) + new) % modulus

    return -1  # Pattern not found in the text.