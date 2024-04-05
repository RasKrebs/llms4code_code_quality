class RabinKarp:
    """
    Rabin-Karp Algorithm for string matching.
    """

    def __init__(self, base: int = 256, prime: int = 101):
        """
        Initialize the base and prime values for hash calculations.
        """
        self.base = base
        self.prime = prime

    def calculate_hash(self, text: str, end: int) -> int:
        """
        Calculate the hash value for a string.
        """
        hash_val = 0
        for i in range(end):
            hash_val = (self.base * hash_val + ord(text[i])) % self.prime
        return hash_val

    def recalculate_hash(self, text: str, old_index: int, new_index: int, old_hash: int, pattern_len: int) -> int:
        """
        Recalculate the hash value after rolling.
        """
        new_hash = ((old_hash - ord(text[old_index]) * pow(self.base, pattern_len - 1)) * self.base + ord(text[new_index])) % self.prime
        return new_hash

    def search_pattern(self, text: str, pattern: str) -> int:
        """
        Search for a pattern in a text string using Rabin-Karp algorithm.
        """
        if not isinstance(text, str) or not isinstance(pattern, str):
            raise ValueError("Both text and pattern should be strings")

        text_len = len(text)
        pattern_len = len(pattern)

        if text_len < pattern_len:
            return -1

        pattern_hash = self.calculate_hash(pattern, pattern_len)
        text_hash = self.calculate_hash(text, pattern_len)

        for i in range(text_len - pattern_len + 1):
            if pattern_hash == text_hash and text[i:i + pattern_len] == pattern:
                return i

            if i < text_len - pattern_len:
                text_hash = self.recalculate_hash(text, i, i + pattern_len, text_hash, pattern_len)

        return -1
