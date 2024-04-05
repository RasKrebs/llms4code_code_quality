class RabinKarp:
    def __init__(self, text: str, pattern: str, base: int = 256, prime: int = 101):
        self.text = text
        self.pattern = pattern
        self.base = base
        self.prime = prime

    def search(self) -> List[int]:
        """
        Search for the pattern in the text using Rabin-Karp algorithm.

        Returns:
        List of indices where the pattern is found in the text.
        """
        results = []
        m = len(self.pattern)
        n = len(self.text)
        pattern_hash = self.hash(self.pattern, m)
        text_hash = self.hash(self.text[:m], m)

        for i in range(n - m + 1):
            if pattern_hash == text_hash and self.text[i:i+m] == self.pattern:
                results.append(i)
            if i < n - m:
                text_hash = self.recalculate_hash(text_hash, self.text[i], self.text[i+m], m)
        
        return results

    def hash(self, s: str, length: int) -> int:
        """
        Calculate the hash value of a string.

        Args:
        s: The string to calculate the hash for.
        length: Length of the string.

        Returns:
        Hash value of the string.
        """
        hash_value = 0
        for char in s:
            hash_value = (hash_value * self.base + ord(char)) % self.prime
        return hash_value

    def recalculate_hash(self, old_hash: int, old_char: str, new_char: str, length: int) -> int:
        """
        Recalculate the hash value based on the old hash, old character, new character and length.

        Args:
        old_hash: Hash value of the old string.
        old_char: Character being removed.
        new_char: Character being added.
        length: Length of the string.

        Returns:
        Recalculated hash value.
        """
        old_hash = (old_hash - ord(old_char) * (self.base**(length-1))) % self.prime
        new_hash = (old_hash * self.base + ord(new_char)) % self.prime
        return new_hash