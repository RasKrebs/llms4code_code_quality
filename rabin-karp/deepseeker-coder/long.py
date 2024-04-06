from typing import Optional

class RabinKarp:
    """
    A class to represent the Rabin-Karp algorithm.

    ...

    Attributes
    ----------
    base : int
        base for the hash function
    prime : int
        prime number for the hash function

    Methods
    -------
    search(text: str, pattern: str) -> int:
        Searches for the pattern in the text.
    """

    def __init__(self, base: Optional[int] = 256, prime: Optional[int] = 101):
        """
        Constructs all the necessary attributes for the RabinKarp object.

        Parameters
        ----------
        base : int, optional
            base for the hash function (default is 256)
        prime : int, optional
            prime number for the hash function (default is 101)
        """

        self.base = base
        self.prime = prime

    def search(self, text: str, pattern: str) -> int:
        """
        Searches for the pattern in the text.

        Parameters
        ----------
        text : str
            The text to search in
        pattern : str
            The pattern to search for

        Returns
        -------
        int
            The index of the first occurrence of the pattern in the text, or -1 if the pattern is not found
        """

        n = len(text)
        m = len(pattern)
        pattern_hash = 0
        text_hash = 0
        h = 1

        # Calculate h = (base^(m-1)) % prime
        for _ in range(m - 1):
            h = (h * self.base) % self.prime

        # Calculate hash value for pattern and text
        for i in range(m):
            pattern_hash = (self.base * pattern_hash + ord(pattern[i])) % self.prime
            text_hash = (self.base * text_hash + ord(text[i])) % self.prime

        # Slide the pattern over text one by one
        for i in range(n - m + 1):
            # Check the hash values of current window of text and pattern
            if pattern_hash == text_hash:
                # Check for characters one by one
                for j in range(m):
                    if text[i + j] != pattern[j]:
                        break
                j += 1
                # If pattern_hash == text_hash and pattern[0...m-1] = text[i, i+1, ...i+m-1]
                if j == m:
                    return i

            # Calculate hash value for next window of text: Remove leading digit, add trailing digit
            if i < n - m:
                text_hash = (self.base * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % self.prime

                # We might get negative values of text_hash, converting it to positive
                if text_hash < 0:
                    text_hash = text_hash + self.prime

        # If we reach here, then pattern is not present in text
        return -1
