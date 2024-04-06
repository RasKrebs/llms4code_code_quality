import heapq
import os
from collections import defaultdict

def calculate_frequency(message):
    frequency = defaultdict(int)
    for symbol in message:
        frequency[symbol] += 1
    return frequency

def build_huffman_tree(frequency):
    heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def huffman_encoding(message):
    frequency = calculate_frequency(message)
    huffman_tree = build_huffman_tree(frequency)
    huffman_codes = {symbol: code for symbol, code in huffman_tree}
    encoded_message = ''.join(huffman_codes[symbol] for symbol in message)
    return encoded_message, huffman_codes

def huffman_decoding(encoded_message, huffman_codes):
    reverse_huffman_codes = {v: k for k, v in huffman_codes.items()}
    decoded_message = ''
    while encoded_message:
        for code in reverse_huffman_codes:
            if encoded_message.startswith(code):
                decoded_message += reverse_huffman_codes[code]
                encoded_message = encoded_message[len(code):]
    return decoded_message
