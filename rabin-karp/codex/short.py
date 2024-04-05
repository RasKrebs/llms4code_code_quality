def rabin_karp(text, pattern):
    pattern_len = len(pattern)
    text_len = len(text)
    pattern_hash = 0
    text_hash = 0
    window_len = pattern_len
    d = 256  # number of characters in the input alphabet
    h = pow(d, pattern_len-1) % text_len  # precompute d^(m-1) % q

    # calculate the hash value of pattern and first window of text
    for i in range(0, pattern_len):
        pattern_hash = (d * pattern_hash + ord(pattern[i])) % text_len
        text_hash = (d * text_hash + ord(text[i])) % text_len

    # slide the pattern over text one by one
    for i in range(0, text_len-pattern_len+1):
        # check the hash values of current window of text and pattern
        # if the hash values match then only check for characters one by one
        if pattern_hash == text_hash:
            for j in range(0, pattern_len):
                if text[i+j] != pattern[j]:
                    break
            j += 1
            # if pattern_hash == text_hash and pattern[0...m-1] = text[i, i+1, ...i+m-1]
            if j == pattern_len:
                return i

        # calculate hash value for next window of text: remove leading digit, add trailing digit
        if i < text_len-pattern_len:
            text_hash = (d * (text_hash - ord(text[i])*h) + ord(text[i+pattern_len])) % text_len

            # we might get negative value of text_hash, converting it to positive
            if text_hash < 0:
                text_hash = text_hash + text_len

    return -1  # pattern not found in the text
