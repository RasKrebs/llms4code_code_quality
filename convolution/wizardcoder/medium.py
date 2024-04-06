import numpy as np
from typing import Tuple, Union

class Convolution2D:
    """A 2-dimensional convolution operation."""

    def __init__(self, padding: int = 0, stride: int = 1, dilation: int = 1):
        """Initialize the convolution.

        Args:
            padding (int): Padding size. Default is 0.
            stride (int): Stride length. Default is 1.
            dilation (int): Dilation rate. Default is 1.
        """
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def run(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply the convolution operation to an image using a specific kernel.

        Args:
            image (np.ndarray): The input image. Must be 2D and real-valued.
            kernel (np.ndarray): The filter kernel. Must be 2D, square and real-valued.

        Returns:
            np.ndarray: The convolved image of same size as the input.
        """
        if self.padding != 0:
            image = self._apply_zero_padding(image, self.padding)

        output_shape = self._get_output_shape(image.shape, kernel.shape, self.stride, self.dilation)
        convolved_image = np.zeros(output_shape)

        for i in range(0, image.shape[0] - kernel.shape[0] + 1, self.stride):
            for j in range(0, image.shape[1] - kernel.shape[1] + 1, self.stride):
                convolved_image[i, j] = np.sum(image[i: i + kernel.shape[0], j: j + kernel.shape[1]] * kernel)

        return convolved_image

    def _apply_zero_padding(self, image: np.ndarray, padding: int):
        """Pad the input image with zeroes on all sides."""
        padded_img = np.zeros((image.shape[0] + 2*padding, image.shape[1] + 2*padding))
        padded_img[padding:-padding, padding:-padding] = image
        return padded_img

    def _get_output_shape(self, image_shape: Tuple[int, int], kernel_shape: Tuple[int, int], stride: int, dilation: int) -> Tuple[int, int]:
        """Calculate the output shape of the convolution operation."""
        out_height = (image_shape[0] - dilation * (kernel_shape[0] - 1) - 1) // stride + 1
        out_width = (image_shape[1] - dilation * (kernel_shape[1] - 1) - 1) // stride + 1

        return out_height, out_width
