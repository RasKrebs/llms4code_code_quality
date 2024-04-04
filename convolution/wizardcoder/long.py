import numpy as np
from typing import Union, Tuple

class Convolution2D:
    """A 2-Dimensional convolution operation."""

    def __init__(self, padding: Union[int, Tuple] = 0, stride: Union[int, Tuple] = 1):
        self.padding = self._check_tuple(padding, 'padding')
        self.stride = self._check_tuple(stride, 'stride')

    def _check_tuple(self, value, name):
        """Check if the input is a tuple or integer and convert it to a tuple."""
        if isinstance(value, int):
            return (value, value)
        elif isinstance(value, tuple):
            assert len(value) == 2, f"{name} should be a 2-tuple."
            assert all(isinstance(i, int) for i in value), f"{name} values should be integers."
            return value
        else:
            raise ValueError(f"Invalid {name}. It should be either an integer or a 2-tuple of integers.")

    def run(self, image: np.ndarray, kernel: Union[np.ndarray, list]):
        """Apply the convolution operation on an image and a kernel."""
        # Convert the list to numpy array if necessary
        if isinstance(kernel, list):
            kernel = np.array(kernel)

        # Check the kernel size and shape
        assert len(kernel.shape) == 2, "Kernel should be a 2D matrix."
        assert kernel.shape[0] == kernel.shape[1], "Kernel should be square."

        pad_width = [(p, p) for p in self.padding]
        image_pad = np.pad(image, pad_width, mode='constant')  # apply padding

        # Perform the convolution
        output_height = (image_pad.shape[0]-kernel.shape[0]) // self.stride[0] + 1
        output_width = (image_pad.shape[1]-kernel.shape[1]) // self.stride[1] + 1

        result = np.zeros((output_height, output_width))  # convolution output

        for i in range(0, output_height):  # loop over rows of the image
            for j in range(0, output_width):  # loop over columns of the image
                h_start = i * self.stride[0]
                h_end = h_start + kernel.shape[0]
                w_start = j * self.stride[1]
                w_end = w_start + kernel.shape[1]

                result[i, j] = np.sum(kernel * image_pad[h_start: h_end, w_start: w_end])  # multiply and sum over the neighborhood

        return result