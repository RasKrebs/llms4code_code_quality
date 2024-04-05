import numpy as np
from typing import Tuple

def convolve2d(image: np.ndarray, kernel: np.ndarray, pad_width: int = 0, stride: int = 1, dilation: int = 1) -> np.ndarray:
    """Perform a valid 2D convolution operation on the input image with the given kernel."""

    # Pad the image based on the padding width
    if pad_width > 0:
        image = np.pad(image, pad_width=pad_width)

    # Compute the output shape (height and width)
    out_height = (image.shape[0] - kernel.shape[0]) // stride + 1
    out_width = (image.shape[1] - kernel.shape[1]) // stride + 1

    # Initialize the output array with zeros
    out_array = np.zeros((out_height, out_width))

    # Perform convolution operation
    for i in range(0, image.shape[0] - kernel.shape[0] + 1, stride):
        for j in range(0, image.shape[1] - kernel.shape[1] + 1, stride):
            window = image[i:i+kernel.shape[0]:dilation, j:j+kernel.shape[1]:dilation]
            out_array[i//stride, j//stride] = np.sum(window * kernel)

    return out_array