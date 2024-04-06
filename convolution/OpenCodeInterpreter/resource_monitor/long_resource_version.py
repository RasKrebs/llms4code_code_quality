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
padding = 1  # Padding of 1 pixel on all sides
stride = 1   # Stride of 1, moving the filter one pixel at a time
dilation = 1 # Dilation of 1, meaning no dilation

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

    def _validate_kernel(self, image: np.ndarray, kernel: Union[np.ndarray, list]) -> np.ndarray:
        kernel = np.array(kernel)
        if len(image.shape) != 2 or len(kernel.shape) != 2:
            raise ValueError('Both image and kernel must be 2D arrays')

        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError('Kernel must be a square matrix')

        # check that the kernel is of correct size for the input image
        output_height = (image.shape[0] + 2 * self._padding[0] - (kernel.shape[0] - 1) * (self._dilation - 1)) // self._stride + 1
        output_width = (image.shape[1] + 2 * self._padding[1] - (kernel.shape[1] - 1) * (self._dilation - 1)) // self._stride + 1

        if not isinstance(output_height, int) or not isinstance(output_width, int):
            raise ValueError('Kernel size is incompatible with the input image size and other parameters')

        return kernel

    def run(self, image: np.ndarray, kernel: Union[np.ndarray, list]) -> np.ndarray:
        # Validate the inputs
        kernel = self._validate_kernel(image, kernel)

        # Calculate the dimensions of the output image
        output_height = int((image.shape[0] + 2 * self._padding[0] - (kernel.shape[0] - 1) * (self._dilation - 1)) / self._stride + 1)
        output_width = int((image.shape[1] + 2 * self._padding[1] - (kernel.shape[1] - 1) * (self._dilation - 1)) / self._stride + 1)

        # Pad the image
        padded_image = np.pad(image, [(self._padding[0], self._padding[0]), (self._padding[1], self._padding[1])], mode='constant', constant_values=0)

        # Create an empty array for the output image
        output = np.zeros((output_height, output_width))

        # Perform convolution operation
        for i in range(output_height):
            for j in range(output_width):
                # Calculate the start and end indices of the image slice
                start_i = i * self._stride
                end_i = start_i + kernel.shape[0]
                start_j = j * self._stride
                end_j = start_j + kernel.shape[1]
                # Multiply corresponding values from image and kernel and sum them up
                try:
                    output[i][j] = np.sum(padded_image[start_i:end_i, start_j:end_j] * kernel)
                except:
                    output[i][j] = 0

        return output

# The execute function
def execute(image, kernel, padding, stride, dilation):
    # Instantiate the Conv2D object with given padding, stride, and dilation
    conv2d = Conv2D(padding, stride, dilation)
    
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

