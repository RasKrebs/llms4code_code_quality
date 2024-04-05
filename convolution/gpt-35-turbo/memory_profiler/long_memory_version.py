from memory_profiler import profile
        import os
        import psutil
        
        # Get the current process ID
        pid = os.getpid()

        # Create a psutil Process object for the current process
        process = psutil.Process(pid)

        # Get the number of logical CPUs in the system
        num_cores = psutil.cpu_count(logical=True)import numpy as np
from typing import Union, Tuple


class Convolution:
    @profile
    def __init__(self, padding: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], dilation: Union[int, Tuple[int, int]]) -> None:
        """
        Initialize the Convolution class with padding, stride, and dilation.

        Args:
        padding (int or tuple): Amount of padding along width and height or total padding.
        stride (int or tuple): Stride value along width and height or total stride.
        dilation (int or tuple): Dilation value along width and height or total dilation.
        """
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

    @profile
    def _pad_image(self, image: np.ndarray) -> np.ndarray:
        """
        Add padding to the input image based on the padding value.

        Args:
        image (np.ndarray): Input image.

        Returns:
        np.ndarray: Padded image.
        """
        # Implement padding based on self.padding
        return padded_image

    @profile
    def _apply_kernel(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Apply the kernel to the image using convolution.

        Args:
        image (np.ndarray): Input image.
        kernel (np.ndarray): Convolutional kernel.

        Returns:
        np.ndarray: Convolved image.
        """
        # Implement convolution between image and kernel
        return convolved_image

    @profile
    def run(self, image: np.ndarray, kernel: Union[np.ndarray, list]) -> np.ndarray:
        """
        Run the convolution operation on the input image with the specified kernel.

        Args:
        image (np.ndarray): Input image.
        kernel (np.ndarray or list): Convolutional kernel.

        Returns:
        np.ndarray: Convolved image.
        """
        if isinstance(kernel, list):
            kernel = np.array(kernel)

        # Include additional checks for edge cases
        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Kernel should be a square matrix.")

        if (image.shape[0] - kernel.shape[0] + 2*self.padding[0]) % self.stride[0] != 0 or \
           (image.shape[1] - kernel.shape[1] + 2*self.padding[1]) % self.stride[1] != 0:
            raise ValueError("Kernel size does not match with the image size based on padding and stride.")

        padded_image = self._pad_image(image)
        convolved_image = self._apply_kernel(padded_image, kernel)

        return convolved_image<write execution script for convolution>
