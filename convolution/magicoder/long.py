import numpy as np
from typing import Union, Tuple

class Convolution:
    def __init__(self, padding: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], dilation:
Union[int, Tuple[int, int]]):
        self.padding = (padding, padding) if isinstance(padding, int) else (padding[0], padding[1])
        self.stride = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else (dilation[0], dilation[1])

    def run(self, image: np.ndarray, kernel: Union[list, np.ndarray]):
        # Ensure the image and the kernel are numpy arrays
        image = np.array(image) if not isinstance(image, np.ndarray) else image
        kernel = np.array(kernel) if not isinstance(kernel, np.ndarray) else kernel

        # Add padding to the image
        image_padded = self._pad_image(image)

        # Check that the kernel size matches the image dimensions
        assert kernel.shape[0] == image_padded.shape[2], "Kernel height does not match image height"
        assert kernel.shape[1] == image_padded.shape[3], "Kernel width does not match image width"

        # Apply the convolution
        result = self._convolution(image_padded, kernel)

        return result

    def _pad_image(self, image: np.ndarray):
        h, w, _, _ = image.shape
        padded_image = np.zeros((h + 2 * self.padding[0], w + 2 * self.padding[1]))
        padded_image[self.padding[0]:self.padding[0] + h, self.padding[1]:self.padding[1] + w] = image[:, :, 0, 0]

        return padded_image

    def _convolution(self, image: np.ndarray, kernel: np.ndarray):
        h, w, c, d = image.shape
        kh, kw, _, _ = kernel.shape

        # Calculate the output size
        out_height = (h - kh) // self.stride[0] + 1
        out_width = (w - kw) // self.stride[1] + 1

        # Initialize the output array
        result = np.zeros((out_height, out_width))

        for i in range(0, h - kh + 1, self.stride[0]):
            for j in range(0, w - kw + 1, self.stride[1]):
                output = np.sum(image[i:i+kh, j:j+kw, 0, 0] * kernel[:, :, 0, 0])
                result[i // self.stride[0], j // self.stride[1]] = output

        return result