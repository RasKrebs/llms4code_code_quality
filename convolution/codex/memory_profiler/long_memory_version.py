from memory_profiler import profile
        import os
        import psutil
        
        # Get the current process ID
        pid = os.getpid()

        # Create a psutil Process object for the current process
        process = psutil.Process(pid)

        # Get the number of logical CPUs in the system
        num_cores = psutil.cpu_count(logical=True)import numpy as np

class Convolution2D:
    @profile
    def __init__(self, padding: int, stride: int, dilation: int):
        """
        Initialize the Convolution2D class.

        :param padding: Padding around the image
        :param stride: Stride for the convolution
        :param dilation: Dilation for the convolution
        """
        self.padding = self._handle_input(padding)
        self.stride = self._handle_input(stride)
        self.dilation = self._handle_input(dilation)

    @staticmethod
    @profile
    def _handle_input(value):
        """
        Handle the input parameters.

        :param value: The input value
        :return: A tuple representing the value
        """
        if isinstance(value, int):
            return (value, value)
        elif isinstance(value, tuple) and len(value) == 2:
            return value
        else:
            raise ValueError("Input should be an integer or a tuple of two integers.")

    @profile
    def run(self, image: np.ndarray, kernel: np.ndarray):
        """
        Run the convolution operation.

        :param image: The input image
        :param kernel: The convolution kernel
        :return: The convolved image
        """
        if isinstance(kernel, list):
            kernel = np.array(kernel)

        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Kernel should be a square matrix.")

        if kernel.shape[0] > image.shape[0] or kernel.shape[1] > image.shape[1]:
            raise ValueError("Kernel should not be larger than the image.")

        # Add padding to the image
        padded_image = np.pad(image, self.padding, mode='constant')

        # Calculate the size of the output image
        output_size = ((padded_image.shape[0] - kernel.shape[0]) // self.stride[0] + 1,
                       (padded_image.shape[1] - kernel.shape[1]) // self.stride[1] + 1)

        # Initialize the output image
        output_image = np.zeros(output_size)

        # Perform the convolution operation
        for i in range(0, padded_image.shape[0] - kernel.shape[0] + 1, self.stride[0]):
            for j in range(0, padded_image.shape[1] - kernel.shape[1] + 1, self.stride[1]):
                output_image[i // self.stride[0], j // self.stride[1]] = np.sum(
                    padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel)

        return output_image<write execution script for convolution>
