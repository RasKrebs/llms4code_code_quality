from typing import List, Tuple

class RabinKarp:
    """
    A class to represent the Rabin-Karp algorithm for string pattern matching.

    ...

    Attributes
    ----------
    text : str
        the text string to search in
    pattern : str
        the pattern string to search for
    base : int
        the base to use in the hash function
    prime : int
        the prime number to use in the hash function

    Methods
    -------
    search():
        Searches for the pattern in the text and returns the starting indices of all occurrences.
    """

    def __init__(self, text: str, pattern: str, base: int = 256, prime: int = 101):
        """
        Constructs all the necessary attributes for the RabinKarp object.

        Parameters
        ----------
            text : str
                the text string to search in
            pattern : str
                the pattern string to search for
            base : int, optional
                the base to use in the hash function (default is 256)
            prime : int, optional
                the prime number to use in the hash function (default is 101)
        """

        self.text = text
        self.pattern = pattern
        self.base = base
        self.prime = prime

    def search(self) -> List[int]:
        """
        Searches for the pattern in the text and returns the starting indices of all occurrences.

        Returns
        -------
        list
            a list of starting indices of all occurrences of the pattern in the text
        """

        n = len(self.text)
        m = len(self.pattern)
        pattern_hash = self._hash(self.pattern, m)
        text_hash = self._hash(self.text, m)

        occurrences = []

        for i in range(n - m + 1):
            if pattern_hash == text_hash:
                if self.pattern == self.text[i:i+m]:
                    occurrences.append(i)

            if i < n - m:
                text_hash = self._rehash(self.text, i, i+m, text_hash, m)

        return occurrences

    def _hash(self, string: str, end: int) -> int:
        """
        Calculates the hash value of a substring of the given string.

        Parameters
        ----------
        string : str
            the string to calculate the hash of
        end : int
            the end index of the substring

        Returns
        -------
        int
            the hash value of the substring
        """

        hash_value = 0
        for i in range(end):
            hash_value = (hash_value * self.base + ord(string[i])) % self.prime
        return hash_value

    def _rehash(self, string: str, old_index: int, new_index: int, old_hash: int, length: int) -> int:
        """
        Re-calculates the hash value of a substring of the given string based on the old hash value.

        Parameters
        ----------
        string : str
            the string to calculate the hash of
        old_index : int
            the old start index of the substring
        new_index : int
            the new start index of the substring
        old_hash : int
            the old hash value of the substring
        length : int
            the length of the substring

        Returns
        -------
        int
            the new hash value of the substring
        """

        new_hash = (old_hash - ord(string[old_index]) * pow(self.base, length - 1, self.prime)) % self.prime
        new_hash = (new_hash * self.base + ord(string[new_index])) % self.prime
        return new_hash
