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

class Convolution:
    """
    Class for performing 2D convolution from scratch.
    """

    def __init__(self, padding: int or tuple, stride: int or tuple, dilation: int or tuple):
        """
        Initializes the convolution parameters.

        Args:
            padding: Integer for equal padding all around, or a tuple (h, w) for separate height and width padding.
            stride: Integer for equal stride in both dimensions, or a tuple (h, w) for separate height and width strides.
            dilation: Integer for equal dilation in both dimensions, or a tuple (h, w) for separate height and width dilations.

        Raises:
            ValueError: If any of the inputs are not integers or tuples of integers.
        """

        self._handle_integer_inputs(padding, stride, dilation)
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def _handle_integer_inputs(self, *inputs: int or tuple):
        """
        Converts integer inputs to tuples for consistency.

        Args:
            *inputs: Variable number of inputs (padding, stride, dilation).

        Raises:
            ValueError: If any of the inputs are not integers or tuples of integers.
        """

        for input_ in inputs:
            if isinstance(input_, int):
                input_ = (input_, input_)
            elif not isinstance(input_, tuple) or any(not isinstance(val, int) for val in input_):
                raise ValueError(f"Invalid input: {input_}. It must be either an integer or a tuple of integers.")

    def run(self, image: np.ndarray, kernel: list or np.ndarray) -> np.ndarray:
        """
        Performs 2D convolution on the given image with the given kernel.

        Args:
            image: NumPy array representing the input image.
            kernel: List or NumPy array representing the convolution kernel.

        Returns:
            The convolved image as a NumPy array.

        Raises:
            ValueError: If the kernel is not a square matrix or is not a list or NumPy array.
            ValueError: If the kernel is not the correct size for the image.
        """

        kernel = self._validate_kernel(kernel)
        return self._convolve(image, kernel)

    def _validate_kernel(self, kernel: list or np.ndarray) -> np.ndarray:
        """
        Validates the kernel and converts it to a NumPy array if necessary.

        Args:
            kernel: Kernel to be validated and converted.

        Returns:
            The validated kernel as a NumPy array.

        Raises:
            ValueError: If the kernel is not a square matrix or is not a list or NumPy array.
        """

        if isinstance(kernel, list):
            kernel = np.array(kernel)

        if not isinstance(kernel, np.ndarray) or not np.array_equal(kernel.shape, (kernel.shape[0], kernel.shape[0])):
            raise ValueError("Kernel must be a square matrix.")

        return kernel

    def _convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Performs the core convolution operation.

        Args:
            image: Input image as a NumPy array.
            kernel: Convolution kernel as a NumPy array.

        Returns:
            The convolved image as a NumPy array.
        """

        # Handle padding
        #image = self._pad_image(image)

        # Calculate output shape
        output_height = int(((image.shape[0] - kernel.shape[0] + 2 * self.padding[0]) // self.stride[0]) + 1)
        output_width = int(((image.shape[1] - kernel.shape[1] + 2 * self.padding[1]) // self.stride[1]) + 1)
        output = np.zeros((output_height, output_width))

        # Perform convolution
        for i in range(output_height):
            for j in range(output_width):
                image_slice = image[
                    i * self.stride[0]
                    ]

# The execute function
def execute(image, kernel, padding, stride, dilation):
    # Instantiate the Conv2D object with given padding, stride, and dilation
    conv2d = Convolution(padding, stride, dilation)
    
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

