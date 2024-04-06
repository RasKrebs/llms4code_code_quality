def rabin_karp(text, pattern):
    if not text or not pattern:
        return -1

    def hash_func(s):
        h = 0
        for char in s:
            h = (h * 256 + ord(char)) % q
        return h

    def rehash(prev_hash, prev_char, next_char, length):
        new_hash = (prev_hash - ord(prev_char) * pow(256, length - 1, q)) * 256 + ord(next_char)
        return new_hash % q

    n = len(text)
    m = len(pattern)
    q = 997
    h_pattern = hash_func(pattern)
    h_text = hash_func(text[:m])

    if h_text == h_pattern and text[:m] == pattern:
        return 0

    for i in range(1, n - m + 1):
        h_text = rehash(h_text, text[i - 1], text[i + m - 1], m)
        if h_text == h_pattern and text[i:i+m] == pattern:
            return i

    return -1
