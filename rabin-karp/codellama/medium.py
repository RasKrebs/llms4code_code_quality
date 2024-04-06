def rabin_karp(text, pattern):
    """
    The Rabin-Karp algorithm for string pattern matching.

    Args:
        text (str): the text to search for the pattern in.
        pattern (str): the pattern to find in the text.

    Returns:
        List[int]: the starting positions of all occurrences of the pattern in the text.
        An empty list is returned if the pattern does not occur anywhere in the text.
    """

    # Preprocessing for Rabin-Karp algorithm
    prime = 101  # A prime number (small enough for simplicity)
    base = 256  # The alphabet size
    d = pow(base, len(pattern) - 1, prime)  # The base raised to the power of pattern length minus one modulo the prime.

    def rolling_hash(string):
        """
        Calculate the hash value for a string using Horner's method.

        Args:
            string (str): the input string to compute the hash for.

        Returns:
            int: the computed hash value for the given string.
        """
        h = 0
        for char in string:
            h = (h * base + ord(char)) % prime

        return h  # Compute the hash value of a pattern using Horner's method.

    pattern_hash = rolling_hash(pattern)  # Calculate the hash value for the pattern.

    def find_matches():
        """
        Find all occurrences of the pattern in the text and return their starting positions.

        Returns:
            List[int]: the starting positions of all occurrences of the pattern in the text.
        """
        result = []
        text_hash = rolling_hash(text[:len(pattern)])  # Calculate the hash value for the first len(pattern) characters in text.

        for i in range(len(text) - len(pattern) + 1):
            if pattern_hash != text_hash:
                if i < len(text) - len(pattern):
                    # Calculate the rolling hash using the pre-computed value and Horner's method.
                    text_hash = (base * (text_hash - ord(text[i]) * d) + ord(text[i + len(pattern)])) % prime
                else:  # No more characters left to hash.
                    break
            elif pattern == text[i:i + len(pattern)]:
                result.append(i)  # Append the starting position if we have a match.

        return result

    return find_matches()  # Return the list of starting positions for all occurrences of the pattern in the text.
