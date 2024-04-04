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
    def __init__(self, padding: Union[int, Tuple[int, int]] = 0, stride: Union[int, Tuple[int, int]] = 1, dilation: Union[int, Tuple[int, int]] = 1):
        """
        Initialize the convolution operation with padding, stride and dilation.

        Parameters:
        padding (Union[int, Tuple[int, int]]): Padding to be applied to the image. If int, same padding is applied to all sides.
        stride (Union[int, Tuple[int, int]]): Stride of the convolution operation. If int, same stride is applied to both dimensions.
        dilation (Union[int, Tuple[int, int]]): Dilation of the convolution operation. If int, same dilation is applied to both dimensions.
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
    def run(self, image: np.ndarray, kernel: Union[np.ndarray, list]):
        """
        Run the convolution operation on the given image with the given kernel.

        Parameters:
        image (np.ndarray): The image to be convolved.
        kernel (Union[np.ndarray, list]): The kernel to be used in the convolution.

        Returns:
        np.ndarray: The convolved image.
        """
        # Convert kernel to numpy array if it's a list
        if isinstance(kernel, list):
            kernel = np.array(kernel)

        # Check if kernel is a square matrix
        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Kernel must be a square matrix.")

        # Check if kernel is the correct size for the image
        if kernel.shape[0] > image.shape[0] or kernel.shape[1] > image.shape[1]:
            raise ValueError("Kernel must be smaller than the image.")

        # Add padding to the image
        image = np.pad(image, self.padding, mode='constant')

        # Initialize output image
        output_shape = ((image.shape[0] - kernel.shape[0]) // self.stride[0] + 1,
                        (image.shape[1] - kernel.shape[1]) // self.stride[1] + 1)
        output = np.zeros(output_shape)

        # Perform convolution
        for i in range(0, output.shape[0], self.stride[0]):
            for j in range(0, output.shape[1], self.stride[1]):
                output[i, j] = np.sum(image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)

        return output<write execution script for convolution>
