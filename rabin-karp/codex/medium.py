class RabinKarp:
    """
    Rabin-Karp algorithm for string pattern matching.
    """

    def __init__(self, base: int = 256, prime: int = 101):
        """
        Initialize the base and prime values for hash calculation.
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

    def recalculate_hash(self, text: str, old_index: int, new_index: int, old_hash: int, length: int) -> int:
        """
        Recalculate the hash value when sliding the window.
        """
        new_hash = ((old_hash - ord(text[old_index]) * pow(self.base, length - 1)) * self.base + ord(text[new_index])) % self.prime
        return new_hash

    def pattern_search(self, text: str, pattern: str) -> list:
        """
        Search for a pattern in a text string.
        """
        text_len = len(text)
        pattern_len = len(pattern)
        pattern_hash = self.calculate_hash(pattern, pattern_len)
        text_hash = self.calculate_hash(text, pattern_len)
        found_indexes = []

        for i in range(text_len - pattern_len + 1):
            if pattern_hash == text_hash and text[i:i+pattern_len] == pattern:
                found_indexes.append(i)
            if i < text_len - pattern_len:
                text_hash = self.recalculate_hash(text, i, i + pattern_len, text_hash, pattern_len)
        return found_indexes
