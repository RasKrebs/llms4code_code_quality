class RabinKarp:
    """Rabin-Karp algorithm class"""

    def __init__(self, pattern_str: str, text_str: str, base=256, prime=101):
        """Initialize the pattern and text strings along with default values for base and prime number.

        Parameters:
            pattern_str (str): The string pattern to be searched in a larger string
            text_str (str): The larger string where pattern is being searched
            base (int, optional): Base used by the hash function Defaults to 256.
            prime (int, optional): Prime number for mod operation in hash calculation Defaults to 101.
        """
        self.pattern_str = pattern_str
        self.text_str = text_str
        self.base = base
        self.prime = prime

    def rabin_karp_search(self)-> int:
        """Implementation of the Rabin-Karp algorithm for pattern searching in a larger string.

        Returns:
            int: The index at which the pattern first occurs, -1 if not found.
        """

        pattern_len = len(self.pattern_str)
        text_len = len(self.text_str)
        prime = self.prime

        # Compute the hash value of the pattern and first window of text
        pattern_hash = 0   # Pattern Hash
        text_hash = 0      # Text Hash
        h = 1

        for i in range(-1):
            h = (h * self.base) % prime

        # Calculate the hash values for pattern and first window of text
        for i in range(pattern_len):
            pattern_hash += ord(self.pattern_str[i]) * h
            text_hash += ord(self.text_str[i]) * h

        pattern_hash = pattern_hash % prime
        text_hash = text_hash % prime

        # Slide the pattern over text one by one
        for i in range(text_len - pattern_len + 1):
            if pattern_hash == text_hash:
                return i   # Pattern found at position i

            # Calculate next window hash value
            if i < text_len - pattern_len:
                text_hash = (self.base * (text_hash - ord(self.text_str[i])*h) + ord(self.text[i+pattern_len])) % prime

                if text_hash < 0:
                    text_hash += prime  # Avoid negative hash values

        return -1   # Pattern not found in text