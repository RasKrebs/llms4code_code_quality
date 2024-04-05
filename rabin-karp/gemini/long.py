class RabinKarp:
   """
   Implements the Rabin-Karp algorithm for pattern searching in a text string.
   """

   def __init__(self, base: int = 256, prime: int = 101):
       """
       Initializes the Rabin-Karp object with optional base and prime values for the hash function.

       Args:
           base: The base to use for the rolling hash function (default: 256).
           prime: The prime number for modulo operation in the hash function (default: 101).
       """

       self.base = base
       self.prime = prime

   def _hash(self, string: str) -> int:
       """
       Calculates the hash value of a string using the rolling hash function.

       Args:
           string: The string to hash.

       Returns:
           The hash value of the string.
       """

       hash_value = 0
       for char in string:
           hash_value = (hash_value * self.base + ord(char)) % self.prime
       return hash_value

   def search(self, text: str, pattern: str) -> int:
       """
       Searches for the first occurrence of a pattern in a text string using the Rabin-Karp algorithm.

       Args:
           text: The text string to search in.
           pattern: The pattern string to search for.

       Returns:
           The index of the first occurrence of the pattern in the text, or -1 if the pattern is not found.
       """

       n = len(text)
       m = len(pattern)

       # Handle edge case: pattern longer than text
       if m > n:
           return -1

       pattern_hash = self._hash(pattern)
       text_hash = self._hash(text[:m])  # Hash of first m characters of text

       for i in range(n - m + 1):
           if pattern_hash == text_hash:
               # Potential match, check for character-by-character equality
               if text[i : i + m] == pattern:
                   return i

           # Calculate hash for the next window in text
           if i < n - m:
               text_hash = (
                   (text_hash - ord(text[i]) * self.base**(m - 1)) * self.base + ord(text[i + m])
               ) % self.prime

       return -1  # Pattern not found

