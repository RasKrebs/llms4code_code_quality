def rabin_karp(text, pattern):
    if len(pattern) > len(text):
        return -1

    q = 2**31-1  # large prime number
    d = 256   # number of unique characters in text/pattern
    h = pow(d, len(pattern)-1) % q  # precomputed value: (d^m-1) mod q

    t_hash = 0  # hash for the text substring
    p_hash = 0  # hash for the pattern

    # calculate initial hash values
    for i in range(len(pattern)):
        t_hash = (d*t_hash + ord(text[i])) % q
        p_hash = (d*p_hash + ord(pattern[i])) % q

    # slide pattern over text and check hash values for equality
    for s in range(len(text)-len(pattern)+1):
        if t_hash == p_hash:
            match = True
            for i in range(len(pattern)):
                if text[s+i] != pattern[i]:
                    match = False
                    break
            if match:
                return s  # found a match

        if s < len(text)-len(pattern):
            t_hash = ((d*(t_hash - h*ord(text[s])) + ord(text[s+len(pattern)])) % q)

    return -1  # no match found