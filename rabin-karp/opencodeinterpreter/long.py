class RollingHash:
    def __init__(self, text: str, size: int):
        self.text = text
        self.size = size
        self.hash = 0
        self.window_start = 0

        for i in range(0, size):
            # ord() gets the Unicode code point of a character
            self.hash += (ord(self.text[i]) - 96) * (31 ** (size - i - 1))

    def move_window(self):
        if self.window_start < len(self.text) - self.size:
            # remove the leftmost character from the hash value
            self.hash -= ((ord(self.text[self.window_start]) - 96) * (31 ** (self.size - 1)))
            # slide the window to include the next character on the right
            self.hash += ((ord(self.text[self.window_start + self.size]) - 96) * (31 ** (self.size - 1)))
            self.window_start += 1

    def window_str(self):
        return self.text[self.window_start: self.window_start + self.size]

def rabin_karp(text: str, pattern: str) -> int:
    if len(pattern) > len(text): return -1

    rt = RollingHash(text, len(pattern))
    rh = RollingHash(pattern, len(pattern))

    for i in range(len(text) - len(pattern) + 1):
        if rt.hash == rh.hash:
            if rt.window_str() == pattern: return i

        rt.move_window()

    return -1