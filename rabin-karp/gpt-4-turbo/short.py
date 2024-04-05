def rabin_karp(text, pattern):
    # Prime number for modulus operation to reduce collisions
    prime = 101
    # Base for the polynomial rolling hash function
    d = 256
    
    # Lengths of text and pattern
    n = len(text)
    m = len(pattern)
    
    # Hash values for text and pattern
    hash_text = 0 
    hash_pattern = 0 
    
    # h = (d^(m-1)) % prime
    h = pow(d, m-1, prime)
    
    # Compute initial hash values for pattern and first window of text
    for i in range(m):
        hash_pattern = (d * hash_pattern + ord(pattern[i])) % prime
        hash_text = (d * hash_text + ord(text[i])) % prime
        
    # Slide the pattern over text one by one 
    for i in range(n - m + 1):
        # Check the hash values of current window of text and pattern
        # If the hash values match, then only check for characters one by one
        if hash_pattern == hash_text:
            # Check for characters one by one
            for j in range(m):
                if text[i+j] != pattern[j]:
                    break
            else:
                # Pattern found at index i
                return i
        
        # Calculate hash value for next window of text
        # Remove leading digit, add trailing digit
        if i < n-m:
            hash_text = (d*(hash_text - ord(text[i])*h) + ord(text[i+m])) % prime
            
            # We might get negative value of t, converting it to positive
            if hash_text < 0:
                hash_text = hash_text + prime
                
    # Pattern not found
    return -1