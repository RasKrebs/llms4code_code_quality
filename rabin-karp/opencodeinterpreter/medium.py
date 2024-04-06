class RabinKarp:
    def __init__(self, text_string: str, pattern_string: str, base: int = 256, prime: int = 101):
        """
        Initializes the RabinKarp class with a given text string, pattern string, and optional base and prime numbers.

        Parameters:
            text_string (str): The string in which to find the pattern.
            pattern_string (str): The string that we are searching for within the text string.
            base (int): The base number used in the hash function. Default is 256.
            prime (int): A prime number larger than the length of the pattern string. It is used to compute the hash value and avoid collisions.
        """
        self.text_string = text_string
        self.pattern_string = pattern_string
        self.base = base
        self.prime = prime

    def search(self) -> list:
        """
        Searches the given text string for all occurrences of the pattern string using the Rabin-Karp algorithm.

        Returns:
            matches (list): A list containing all start indices in the text string where a match with the pattern string is found.

        """
        m = len(self.pattern_string)
        n = len(self.text_string)

        # Compute hash values for pattern and first window of text.
        pattern_hash = 0
        text_hash = 0
        h = 1
        matches = []

        for i in range(m-1):
            h = (h*self.base)%self.prime

        for i in range(m):
            pattern_hash = (self.base * pattern_hash + ord(self.pattern_string[i])) % self.prime
            text_hash = (self.base * text_hash + ord(self.text_string[i])) % self.prime

        # Slide the window over the text, one character at a time.
        for i in range(n-m+1):
            if pattern_hash == text_hash:
                j = 0
                while j < m:
                    if self.text_string[i+j] != self.pattern_string[j]:
                        break

                    j += 1

                # If we've matched all characters in the pattern string, add this match to our list of matches.
                if j == m:
                    matches.append(i)

            # Slide the window over one character and recompute the hash for the new window.
            if i < n-m:
                text_hash = (self.base*(text_hash - ord(self.text_string[i])*h) + ord(self.text_string[i+m])) % self.prime

                # Ensure that the hash is a positive number.
                if text_hash < 0:
                    text_hash += self.prime

        return matches
