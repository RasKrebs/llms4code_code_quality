def rabin_karp(text, pattern):
    d = 256  # number of characters in the input alphabet
    q = 101  # a prime number
    m = len(pattern)
    n = len(text)
    p = 0  # hash value for pattern
    t = 0  # hash value for text
    h = 1

    # The value of h would be "pow(d, M-1)%q"
    for i in range(m-1):
        h = (h*d)%q

    # Calculate the hash value of pattern and first window of text
    for i in range(m):
        p = (d*p + ord(pattern[i]))%q
        t = (d*t + ord(text[i]))%q

    # Slide the pattern over text one by one
    for s in range(n-m+1):
        # Check the hash values of current window of text and pattern
        # If the hash values match then only check for characters on by one
        if p == t:
            # Check for characters one by one
            for j in range(m):
                if text[s+j] != pattern[j]:
                    break
            j+=1
            # if p == t and pattern[0...M-1] = txt[i, i+1, ...i+M-1]
            if j == m:
                return s

        # Calculate hash value for next window of text: Remove leading digit, add trailing digit
        if s < n-m:
            t = (d*(t-ord(text[s])*h) + ord(text[s+m]))%q

            # We might get negative values of t, converting it to positive
            if t < 0:
                t = t+q
    return -1