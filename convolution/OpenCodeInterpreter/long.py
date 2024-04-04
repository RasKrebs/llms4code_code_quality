import numpy as np
from typing import Union, Tuple

class Conv2D():
    """
    A class that represents a 2D convolution operation.

    Attributes
    ----------
    padding : tuple
        Padding for the image
    stride : int
        Stride of the kernel over input image
    dilation : int
        Dilation rate to increase the receptive field

    Methods
    -------
    __init__(padding, stride, dilation) -> None:
        Initializes the convolution operation with padding, stride and dilation.

    _validate_kernel(image, kernel):
        Validates if the given kernel is a square matrix of correct size for the image.

    run(image, kernel):
        Performs the 2D convolution operation on the input image with the given kernel.
    """

    def __init__(self, padding: Union[int, Tuple[int, int]], stride: int, dilation: int) -> None:
        self._padding = (padding, padding) if isinstance(padding, int) else padding
        self._stride = stride
        self._dilation = dilation

    def _validate_kernel(self, image: np.ndarray, kernel: Union[np.ndarray, list]) -> None:
        kernel = np.array(kernel)
        if len(image.shape) != 2 or len(kernel.shape) != 2:
            raise ValueError('Both image and kernel must be 2D arrays')

        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError('Kernel must be a square matrix')

        # check that the kernel is of correct size for the input image
        output_height = (image.shape[0] + 2 * self._padding - (kernel.shape[0] - 1) * (self._dilation - 1)) /self._stride + 1
        output_width = (image.shape[1] + 2 * self._padding - (kernel.shape[1] - 1) * (self._dilation - 1)) / self._stride + 1

        if not output_height.is_integer() or not output_width.is_integer():
            raise ValueError('Kernel size is incompatible with the input image size and other parameters')

        return kernel

    def run(self, image: np.ndarray, kernel: Union[np.ndarray, list]) -> np.ndarray:

        # check that inputs are valid
        kernel = self._validate_kernel(image, kernel)

        output_height = int((image.shape[0] + 2 * self._padding - (kernel.shape[0] - 1) * (self._dilation - 1)) /
self._stride + 1)
        output_width = int((image.shape[1] + 2 * self._padding - (kernel.shape[1] - 1) * (self._dilation - 1)) /
self._stride + 1)

        # create an empty array for the output image
        output = np.zeros((output_height, output_width))

        # perform convolution operation
        for i in range(output_height):
            for j in range(output_width):
                # multiply corresponding values from image and kernel and sum them up
                output[i][j] = np.sum(image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)

        return output