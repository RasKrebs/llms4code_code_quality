import psutil
import os
import threading
import multiprocessing

# Define a global variable to store the maximum resources usage
max_resources_usage = {"cpu": 0, "memory": 0}



# Define a simple 5x5 image
import numpy as np

np.random.seed(0)

# Define the shape of the image
image_shape = (1980, 1280)

# Create a random image array
image = np.random.randint(0, 256, size=image_shape)

# Define a simple 3x3 kernel
kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Specify padding, stride, and dilation
padding = (1, 1)  # Padding of 1 pixel on all sides
stride = (1, 1)   # Stride of 1, moving the filter one pixel at a time
dilation = (1, 1) # Dilation of 1, meaning no dilation


# Implement the resource monitor
def resource_monitor():
    """
    Monitors the CPU and memory usage of the current process, updating global max usage.
    """
    global max_resources_usage
    process = psutil.Process(os.getpid())
    
    while monitoring:
        cpu_usage = process.cpu_percent(interval=1) / multiprocessing.cpu_count()
        memory_usage = process.memory_info().rss
        max_resources_usage['cpu'] = max(max_resources_usage['cpu'], cpu_usage)
        max_resources_usage['memory'] = max(max_resources_usage['memory'], memory_usage)



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

# The execute function
def execute(image, kernel, padding, stride, dilation):
    # Instantiate the Conv2D object with given padding, stride, and dilation
    conv2d = Convolution2D(padding[0], stride[0], dilation[0])
    
    # Run the convolution process
    convolved_image = conv2d.run(image, kernel)
    
    return convolved_image



if __name__ == "__main__":
    # Start the resource monitoring in a separate thread
    global monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=resource_monitor)
    monitor_thread.start()

    # Execute the Huffman coding process
    con = execute(image, kernel, padding, stride, dilation)


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

