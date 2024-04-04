from memory_profiler import profile
        import os
        import psutil
        
        # Get the current process ID
        pid = os.getpid()

        # Create a psutil Process object for the current process
        process = psutil.Process(pid)

        # Get the number of logical CPUs in the system
        num_cores = psutil.cpu_count(logical=True)import numpy as np
from typing import Union, List, Tuple

class Conv2D():
    """
    Implement a class to perform 2D-convolution on images.

    Arguments:
        padding (Union[int, Tuple]): padding in integer or tuple form for both width and height
            e.g. (1, 3) - padding of 1 pixel in the vertical direction and 3 pixels in horizontal
                5      - same padding value of 5 pixels for both dimensions
        stride (Union[int, Tuple]): stride in integer or tuple form for both width and height
            e.g. (2, 4) - stride of 2 in vertical direction and 4 in horizontal
                10      - same stride value of 10 pixels for both dimensions
        dilation (Union[int, Tuple]): dilation in integer or tuple form for both width and height
            e.g. (2, 3) - dilation of 2 in vertical direction and 3 in horizontal
                5      - same dilation value of 5 pixels for both dimensions
    """
    @profile
    def __init__(self, padding: Union[int, Tuple], stride: Union[int, Tuple], dilation: Union[int, Tuple]):
        # Handle the case where integer values are provided as input
        if isinstance(padding, int) and padding >= 0:
            self.padding = (padding, padding)
        else:
            raise TypeError("Padding should be a non-negative integer or tuple of integers.")

        if isinstance(stride, int) and stride > 0:
            self.stride = (stride, stride)
        else:
            raise TypeError("Stride should be a positive integer or tuple of positive integers.")

        if isinstance(dilation, int) and dilation > 0:
            self.dilation = (dilation, dilation)
        else:
            raise TypeError("Dilation should be a positive integer or tuple of positive integers.")

    @profile
    def __validate_image(self, image: np.ndarray):
        """
        Ensure that the input image is a 2D NumPy array.

        Arguments:
            image (np.ndarray): Input image to be validated

        Raises:
            TypeError: if 'image' is not a 2D NumPy array
        """
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise TypeError("Image must be a 2D NumPy array.")

    @profile
    def __validate_kernel(self, kernel: Union[np.ndarray, List]):
        """
        Ensure that the input kernel is a square matrix (2D) and convert it to a NumPy array if necessary.

        Arguments:
            kernel (Union[np.ndarray, List]): Input kernel to be validated

        Raises:
            TypeError: if 'kernel' is not a list or 2D NumPy array
            ValueError: if the input kernel is not a square matrix

        Returns:
            (np.ndarray) Validated and converted NumPy array of the kernel.
        """
        if isinstance(kernel, np.ndarray):
            pass
        elif isinstance(kernel, list) and all([isinstance(row, list) for row in kernel]):
            # Convert the list to a NumPy array if it's not already
            kernel = np.asarray(kernel)
        else:
            raise TypeError("Kernel must be a 2D NumPy array or list of lists.")

        # Check that the kernel is square (i.e., has equal width and height)
        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Kernel must be a square matrix (equal width and height).")

        return kernel

    @profile
    def __apply_padding(self, image: np.ndarray):
        """
        Apply padding to the input image.

        Arguments:
            image (np.ndarray): Input image to be padded

        Returns:
            (np.ndarray) Padded image.
        """
        pad_width = ((self.padding[0], self.padding[0]), (self.padding[1], self.padding[1]))
        return np.pad(image, pad_width=pad_width, mode='constant')

    @profile
    def __apply_dilation(self, kernel: np.ndarray):
        """
        Apply dilation to the input kernel.

        Arguments:
            kernel (np.ndarray): Input kernel to be dilated

        Returns:
            (np.ndarray) Dilated kernel.
        """
        return np.ones(kernel.shape + 2 * self.dilation, dtype=kernel.dtype) * kernel[:, :, None, None]

    @profile
    def run(self, image: np.ndarray, kernel: Union[np.ndarray, List]) -> np.ndarray:
        """
        Perform convolution on an input image with a given kernel.

        Arguments:
            image (np.ndarray): Input image to be convolved
            kernel (Union[np.ndarray, List]): Convolutional kernel to apply on the image

        Returns:
            (np.ndarray) Convolved image.
        """
        # Validate the input image and kernel
        self.__validate_image(image)
        kernel = self.__validate_kernel(kernel)

        # Apply padding to the input image if necessary
        padded_image = self.__apply_padding(image)

        # Apply dilation to the kernel if necessary
        dilated_kernel = self.__apply_dilation(kernel)

        # Initialize output convolved image
        conv_image = np.zeros((padded_image.shape[0] - dilated_kernel.shape[0], padded_image.shape[1] -
dilated_kernel.shape[1]))

        # Perform convolution on the input image with the kernel using NumPy's 'convolve2d' function
        return np.convolve2d(padded_image, np.rot90(dilated_kernel, k=2), mode='valid')<write execution script for convolution>
