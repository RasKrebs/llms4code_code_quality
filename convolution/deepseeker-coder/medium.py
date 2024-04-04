import numpy as np
from typing import Tuple

class Convolution:
    """
    A class to perform convolution operation on an image with a given kernel.
    """

    def __init__(self, padding: int = 0, stride: int = 1, dilation: int = 1):
        """
        Initialize the convolution operation with padding, stride and dilation.

        Parameters:
        padding (int): The number of pixels to pad the image with. Default is 0.
        stride (int): The number of pixels to skip when moving the kernel. Default is 1.
        dilation (int): The number of pixels inserted between kernel elements. Default is 1.
        """
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def run(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Perform convolution on the given image with the given kernel.

        Parameters:
        image (np.ndarray): The image to convolve.
        kernel (np.ndarray): The kernel to convolve with.

        Returns:
        np.ndarray: The convolved image.
        """
        # Add padding to the image
        image_padded = np.pad(image, self.padding, mode='constant')

        # Get the dimensions of the image and kernel
        image_height, image_width = image_padded.shape
        kernel_height, kernel_width = kernel.shape

        # Calculate the output dimensions
        output_height = (image_height - kernel_height) // self.stride + 1
        output_width = (image_width - kernel_width) // self.stride + 1

        # Initialize the output image
        output = np.zeros((output_height, output_width))

        # Perform convolution
        for i in range(0, output_height):
            for j in range(0, output_width):
                # Get the current region of the image
                image_region = image_padded[i * self.stride:i * self.stride + kernel_height,
                                            j * self.stride:j * self.stride + kernel_width]

                # Perform element-wise multiplication and summation
                output[i, j] = np.sum(image_region * kernel)

        return output