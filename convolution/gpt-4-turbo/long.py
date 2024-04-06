import numpy as np

class Convolution2D:
    """
    A class that performs 2D convolution on an image.
    
    Attributes:
        padding (tuple): padding around the image.
        stride (tuple): the stride of the convolution.
        dilation (tuple): the dilation of the kernel.
    
    Methods:
        run(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
            Performs the convolution operation on the given image with the provided kernel.
    """

    def __init__(self, padding: int | tuple, stride: int | tuple, dilation: int | tuple):
        """
        Initializes the Convolution2D class with padding, stride, and dilation.

        Parameters:
            padding (int | tuple): The padding to apply on all sides of the image, as either an int or a tuple of (height padding, width padding).
            stride (int | tuple): The stride of the convolution, as either an int or a tuple of (height stride, width stride).
            dilation (int | tuple): The dilation applied to the kernel, as either an int or a tuple of (height dilation, width dilation).
        
        Raises:
            ValueError: If any of the inputs are not integers or tuples of integers.
        """
        self.padding = self._validate_and_convert_input(padding)
        self.stride = self._validate_and_convert_input(stride)
        self.dilation = self._validate_and_convert_input(dilation)

    def _validate_and_convert_input(self, input_value: int | tuple) -> tuple:
        """
        Validates and converts the input to a tuple if it is an integer.

        Parameters:
            input_value (int | tuple): The input to be validated and converted.

        Returns:
            tuple: A tuple representing the input value.

        Raises:
            ValueError: If the input is not an integer or a tuple of integers.
        """
        if isinstance(input_value, int):
            return (input_value, input_value)
        elif isinstance(input_value, tuple) and all(isinstance(val, int) for val in input_value):
            return input_value
        else:
            raise ValueError("Input must be an integer or a tuple of integers")

    @staticmethod
    def _to_numpy_array(matrix):
        """
        Converts a list to a numpy array if not already one.

        Parameters:
            matrix: The matrix to be converted.

        Returns:
            numpy.ndarray: The given matrix as a numpy array.
        """
        if isinstance(matrix, list):
            return np.array(matrix)
        elif isinstance(matrix, np.ndarray):
            return matrix
        else:
            raise ValueError("Kernel should be a list or a numpy array.")

    def run(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Performs the convolution operation on the input image with the given kernel.

        Parameters:
            image (np.ndarray): The input image.
            kernel (np.ndarray): The convolution kernel.

        Returns:
            np.ndarray: The convolved image.

        Raises:
            ValueError: If kernel is not a square matrix, or if it has incorrect size for the image.
        """
        kernel = self._to_numpy_array(kernel)

        # Validate kernel shape
        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Kernel must be a square matrix.")

        # Padding the image
        image_padded = np.pad(image, ((self.padding[0],), (self.padding[1],)), mode='constant')

        # Calculate output dimensions
        output_height = ((image.shape[0] + 2 * self.padding[0] - kernel.shape[0]) // self.stride[0]) + 1
        output_width = ((image.shape[1] + 2 * self.padding[1] - kernel.shape[1]) // self.stride[1]) + 1

        # Create an empty array to store the output
        output = np.zeros((output_height, output_width))

        # Perform convolution
        for y in range(0, output_height):
            for x in range(0, output_width):
                output[y, x] = np.sum(
                    image_padded[
                        y * self.stride[0]:y * self.stride[0] + kernel.shape[0],
                        x * self.stride[1]:x * self.stride[1] + kernel.shape[1]
                    ] * kernel
                )

        return output
