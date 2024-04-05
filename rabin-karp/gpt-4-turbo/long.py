class RabinKarp:
    """
    Rabin-Karp Algorithm implementation for pattern matching.
    This class implements the Rabin-Karp Algorithm, which utilizes a rolling
    hash function for efficient substring search. The algorithm matches the
    hash value of the pattern with substring hash values of the text.
    """

    def __init__(self, base: int = 256, prime: int = 101):
        """
        Initialize the Rabin-Karp class with optional base and prime numbers.

        :param base: The base value for the rolling hash function. Default is 256.
        :param prime: The prime number to modulate the hash. Default is 101.
        """
        self.base = base
        self.prime = prime

    def search(self, text: str, pattern: str) -> int:
        """
        Search for the pattern in the given text using the Rabin-Karp algorithm.

        :param text: The text in which to search for the pattern.
        :param pattern: The pattern to search for.
        :return: The index of the first occurrence of the pattern in the text, or -1 if not found.
        """
        n = len(text)
        m = len(pattern)
        if m > n:
            return -1
        
        pattern_hash = 0
        text_hash = 0
        h = 1  # The value for the highest digit's contribution to hash

        # Pre-processing: Calculate the hash value of the pattern and the first window of the text
        for i in range(m):
            if i > 0:
                h = (h * self.base) % self.prime
            pattern_hash = (self.base * pattern_hash + ord(pattern[i])) % self.prime
            text_hash = (self.base * text_hash + ord(text[i])) % self.prime

        # Slide the pattern over text one by one and check for matches
        for i in range(n - m + 1):
            if pattern_hash == text_hash:
                # Check characters one by one for a sure match
                for j in range(m):
                    if text[i + j] != pattern[j]:
                        break
                else:  # Match found
                    return i

            if i < n - m:
                # Calculate hash value for the next window: Remove leading digit, add trailing digit
                text_hash = (self.base * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % self.prime

                # We might get a negative value of text_hash, converting it to positive
                if text_hash < 0:
                    text_hash = text_hash + self.prime

        return -1  # Pattern not found