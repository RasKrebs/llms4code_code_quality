class RollingHash:
    def __init__(self, base=101, prime=257):
        """
        Initialize a new RollingHash instance.

        Args:
            base (int): Base value for the hash function. Defaults to 101.
            prime (int): Prime number for the hash function. Defaults to 257.
        """
        self.base = base
        self.prime = prime

    def get_hash(self, text):
        """
        Calculate the rolling hash value of a string.

        Args:
            text (str): The input string to calculate the hash for.

        Returns:
            int: The rolling hash value for the given string.
        """
        hash_value = 0
        for char in text:
            hash_value = (hash_value * self.base + ord(char)) % self.prime
        return hash_value

    def update_hash(self, old_text, new_text):
        """
        Update the rolling hash value of a string by replacing one character.

        Args:
            old_text (str): The original text for which the hash is already calculated.
            new_text (str): New text with exactly one character different from the original text.

        Returns:
            int: The updated rolling hash value for the new string.
        """
        if len(old_text) != len(new_text):
            raise ValueError("New text must have the same length as old text.")

        # Calculate the difference between the hashes of the two strings
        diff = 0
        for i in range(len(old_text)):
            diff += (ord(old_text[i]) - ord(new_text[i])) * pow(self.base, len(old_text) - i - 1, self.prime)

        # Update the hash by subtracting the difference and taking modulo
        new_hash = (self.get_hash(old_text) - diff) % self.prime
        return new_hash

def rabin_karp_search(text, pattern, base=101, prime=257):
    """
    Search for a pattern string in a text using the Rabin-Karp algorithm.

    Args:
        text (str): The input text to search in.
        pattern (str): The pattern string to search for.
        base (int, optional): Base value for the hash function. Defaults to 101.
        prime (int, optional): Prime number for the hash function. Defaults to 257.

    Returns:
        int: The index of the first occurrence of the pattern in the text, or -1 if not found.
    """
    # Initialize a RollingHash instance with the given base and prime
    rh = RollingHash(base=base, prime=prime)

    # Calculate hash values for pattern and initial substring of text
    pattern_hash = rh.get_hash(pattern)
    text_hash = rh.get_hash(text[:len(pattern)])

    # Check if the first substring matches the pattern
    if pattern_hash == text_hash:
        return 0

    for i in range(1, len(text) - len(pattern) + 1):
        # Update the rolling hash of the next substring by replacing one character
        old_substring = text[i-1:len(pattern)+i-1]
        new_substring = text[i:len(pattern)+i]
        text_hash = rh.update_hash(old_substring, new_substring)

        # Check if the updated substring matches the pattern
        if pattern_hash == text_hash and pattern == new_substring:
            return i

    # If no match is found, return -1
    return -1
