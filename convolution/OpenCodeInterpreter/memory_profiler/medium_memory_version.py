from memory_profiler import profile
        import os
        import psutil
        
        # Get the current process ID
        pid = os.getpid()

        # Create a psutil Process object for the current process
        process = psutil.Process(pid)

        # Get the number of logical CPUs in the system
        num_cores = psutil.cpu_count(logical=True)from typing import Tuple
import numpy as np

class Convolution:
    @profile
    def __init__(self, padding=0, stride=1, dilation=1):
        """ Initialize convolution parameters.

        Parameters
        ----------
        padding : int (default: 0)
            Padding size for input image and kernel

        stride : int (default: 1)
            Stride size for moving the kernel across the image.

        dilation : int (default: 1)
            Dilation size, used to increase the receptive field of the kernel.
        """
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    @profile
    def _add_padding(self, image : np.ndarray, padding: int) -> np.ndarray:
        """ Add padding to the input image.

        Parameters
        ----------
        image : numpy array
            Input image as a numpy array

        padding : int
            Padding size for input image and kernel
        """
        return np.pad(image, (padding, padding))

    @profile
    def _convolve(self, patch: np.ndarray, kernel: np.ndarray) -> float:
        """ Convolves the given patch with the given kernel."""

        return np.sum(patch * kernel)

    @profile
    def run(self, image : np.ndarray, kernel: np.ndarray):
        """ Run convolution operation on an image with a specific kernel.

        Parameters
        ----------
        image : numpy array
            Input image as a numpy array

        kernel : numpy array
            Kernel to be used for the convolution operation.
        """
        padded_image = self._add_padding(image, self.padding)

        output_height = int((padded_image.shape[0] - kernel.shape[0]) / self.stride + 1)
        output_width = int((padded_image.shape[1] - kernel.shape[1]) / self.stride + 1)

        result = np.zeros((output_height, output_width))

        for h in range(0, output_height):
            for w in range(0, output_width):
                vert_start = self.stride * h
                vert_end = vert_start + kernel.shape[0]

                horiz_start = self.stride * w
                horiz_end = horiz_start + kernel.shape[1]

                patch = padded_image[vert_start:vert_end, horiz_start:horiz_end]

                result[h, w] = self._convolve(patch, kernel)

        return result<write execution script for convolution>
