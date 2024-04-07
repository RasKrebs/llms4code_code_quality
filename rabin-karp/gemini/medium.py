class RabinKarp:
  """
  This class implements the Rabin-Karp algorithm for string pattern matching.

  Attributes:
    d: The number of characters in the input alphabet (default: 256).
    q: A prime number used in the hash function (default: 101).
  """

  def __init__(self, d: int = 256, q: int = 101) -> None:
    """
    Initializes the RabinKarp object with the given alphabet size and prime number.

    Args:
      d: The number of characters in the input alphabet (default: 256).
      q: A prime number used in the hash function (default: 101).
    """
    self.d = d
    self.q = q

  def char_to_int(self, char: str) -> int:
    """
    Converts a character to its corresponding integer value.

    Args:
      char: The character to convert.

    Returns:
      The integer value of the character.
    """
    return ord(char)

  def hash(self, text: str, i: int, n: int) -> int:
    """
    Calculates the hash value of a substring of the given text.

    Args:
      text: The text string.
      i: The starting index of the substring.
      n: The length of the substring.

    Returns:
      The hash value of the substring.
    """
    h = 0
    for j in range(n):
      h = (h * self.d + self.char_to_int(text[i + j])) % self.q
    return h

  def search(self, text: str, pattern: str) -> list[int]:
    """
    Searches for the pattern in the text using the Rabin-Karp algorithm.

    Args:
      text: The text string.
      pattern: The pattern string to search for.

    Returns:
      A list containing the starting indices of all occurrences of the pattern in the text, or an empty list if not found.
    """

    n = len(pattern)
    m = len(text)

    d = self.d
    q = self.q

    p = self.hash(pattern, 0, n)  # hash value for the pattern
    h = 0  # hash value for the current substring

    x = 1  # pow(d, n-1) % q

    if n > m:
      return []

    # Preprocessing
    for i in range(n - 1):
      x = (x * d) % q

    # Calculate hash value for the first window
    for i in range(n):
      h = (h * d + self.char_to_int(text[i])) % q

    result = []

    # Slide the pattern over the text
    for i in range(m - n + 1):
      # Check if the hash values match
      if p == h:
        # If hash values match, check if characters match
        if text[i:i + n] == pattern:
          result.append(i)

      # Calculate hash value for the next window
      if i < m - n:
        h = (d * (h - self.char_to_int(text[i]) * x) + self.char_to_int(text[i + n])) % q
        h = (h + q) % q  # handle negative hash value

    return result
