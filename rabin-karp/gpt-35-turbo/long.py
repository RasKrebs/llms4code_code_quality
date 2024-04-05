class RabinKarpAlgorithm:
    def __init__(self, text: str, pattern: str, base: int = 256, prime: int = 101):
        self.text = text
        self.pattern = pattern
        self.base = base
        self.prime = prime
        self.text_len = len(text)
        self.pattern_len = len(pattern)
        self.base_power = pow(base, pattern_len - 1, prime)
        self.text_hash = self.hash_text(text)
        self.pattern_hash = self.hash_pattern(pattern)

    def hash_text(self, text: str) -> int:
        hash_value = 0
        for char in text:
            hash_value = (hash_value * self.base + ord(char)) % self.prime
        return hash_value

    def hash_pattern(self, pattern: str) -> int:
        hash_value = 0
        for char in pattern:
            hash_value = (hash_value * self.base + ord(char)) % self.prime
        return hash_value

    def check_equal_strings(self, text: str, start: int) -> bool:
        return text == self.text[start:start + self.pattern_len]

    def search(self) -> int:
        if self.pattern_len > self.text_len:
            return -1

        if self.pattern_hash == self.text_hash and self.check_equal_strings(self.text, 0):
            return 0

        for i in range(1, self.text_len - self.pattern_len + 1):
            # Recalculate hash value using rolling hash
            self.text_hash = (self.base * (self.text_hash - ord(self.text[i - 1]) * self.base_power) + ord(self.text[i + self.pattern_len - 1])) % self.prime

            # Check if hash values are equal and then verify with actual strings
            if self.text_hash == self.pattern_hash and self.check_equal_strings(self.text, i):
                return i

        return -1