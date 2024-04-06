class RabinKarp:
    def __init__(self, text: str, pattern: str, d: int = 256, q: int = 101):
        """
        Initialize the instance of class with given parameters.

        Parameters:
            text (str): The input string.
            pattern (str): The pattern to search in the text.
            d (int): A prime number for modulo operation. Defaults to 256.
            q (int): A base number to use in the hash function. Defaults to 101.
        """
        self.text = text
        self.pattern = pattern
        self.d = d
        self.q = q

    def _compute_hash(self, string: str) -> int:
        """
        Compute hash value of a given string.

        Parameters:
            string (str): The input string for which the hash needs to be computed.

        Returns:
            int: The hash value of the string.
        """
        length = len(string)
        h = 0
        for i in range(length):
            h = (h + ord(string[i]) * self.d ** (length - i - 1)) % self.q
        return h

    def _search(self) -> list:
        """
        Search the pattern in text using Rabin-Karp algorithm.

        Returns:
            list[int]: List of indices where pattern is found. Empty if not found.
        """
        M = len(self.pattern)
        N = len(self.text)
        p = 0    # hash value for pattern
        t = 0    # hash value for text
        h = 1
        result = []

        for _ in range(M-1):
            h = (h * self.d) % self.q

        # Calculate the hash of pattern and first window of text
        p = self._compute_hash(self.pattern)
        t = self._compute_hash(self.text[0:M])

        for i in range(N-M+1):
            if p == t:  # Check the hash values
                for j in range(M):
                    if self.text[i + j] != self.pattern[j]:    # character comparison
                        break

                j += 1
                if j == M:
                    result.append(i)  # Add the index to result list

            if i < N-M:
                t = (self.d * (t - ord(self.text[i]) * h) + ord(self.text[i + M])) % self.q

        return result

    def search(self):
        """
        This is the main function that calls _search() to find pattern in text using Rabin-Karp algorithm.

        Returns:
            list[int]: List of indices where pattern is found. Empty if not found.
        """
        return self._search()
