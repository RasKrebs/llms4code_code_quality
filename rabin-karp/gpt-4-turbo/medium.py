class RabinKarp:
    """
    Rabin-Karp Algorithm class for string pattern matching.

    Attributes:
        text (str): The text string to be searched.
        pattern (str): The pattern string to be found in the text.
        base (int): Base value for hashing function. Default is 256 (number of ASCII characters).
        prime (int): A prime number to ensure a more uniform distribution of hash values. 
    """

    def __init__(self, text: str, pattern: str, base: int = 256, prime: int = 101):
        """
        Initializes the Rabin-Karp algorithm with the text and pattern strings,
        along with custom values for the base and prime numbers used in the hash function.
        """
        self.text = text
        self.pattern = pattern
        self.base = base
        self.prime = prime

    def search(self) -> list[int]:
        """
        Searches for all occurrences of the pattern in the text.

        Returns:
            A list of starting indices of all occurrences of the pattern in the text.
        """
        result = []  # To store the starting indexes of matched patterns
        M, N = len(self.pattern), len(self.text)
        i, j, p, t = 0, 0, 0, 0  # Initialize variables for pattern, text hash values and indexes
        h = 1  # The value of h would be "pow(d, M-1)%q"

        # Calculate the h value (pow(d, M-1) % q)
        for i in range(M-1):
            h = (h * self.base) % self.prime

        # Calculate the hash value of pattern and first window of text
        for i in range(M):
            p = (self.base * p + ord(self.pattern[i])) % self.prime
            t = (self.base * t + ord(self.text[i])) % self.prime

        # Slide the pattern over text one by one
        for i in range(N - M + 1):
            # Check the hash values of current window of text and pattern.
            # If the hash values match then only check for characters one by one
            if p == t:
                # Check for characters one by one
                for j in range(M):
                    if self.text[i + j] != self.pattern[j]:
                        break
                j += 1
                # if p == t and pattern[0...M-1] = text[i, i+1, ...i+M-1]
                if j == M:
                    result.append(i)

            # Calculate hash value for next window of text: Remove leading digit,
            # add trailing digit
            if i < N - M:
                t = (self.base * (t - ord(self.text[i]) * h) + ord(self.text[i + M])) % self.prime

                # We might get negative value of t, converting it to positive
                if t < 0:
                    t = t + self.prime

        return result