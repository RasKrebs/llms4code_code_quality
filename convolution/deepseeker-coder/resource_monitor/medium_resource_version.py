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
from typing import Tuple

class Convolution:
    """
    A class to perform convolution operation on an image with a given kernel.
    """

    def __init__(self, padding: int = 0, stride: int = 1, dilation: int = 1):
        """
        Initialize the convolution operation with padding, stride and dilation.

        Parameters:
        padding (int): The number of pixels to pad the image with. Default is 0.
        stride (int): The number of pixels to skip when moving the kernel. Default is 1.
        dilation (int): The number of pixels inserted between kernel elements. Default is 1.
        """
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def run(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Perform convolution on the given image with the given kernel.

        Parameters:
        image (np.ndarray): The image to convolve.
        kernel (np.ndarray): The kernel to convolve with.

        Returns:
        np.ndarray: The convolved image.
        """
        # Add padding to the image
        image_padded = np.pad(image, self.padding, mode='constant')

        # Get the dimensions of the image and kernel
        image_height, image_width = image_padded.shape
        kernel_height, kernel_width = kernel.shape

        # Calculate the output dimensions
        output_height = (image_height - kernel_height) // self.stride + 1
        output_width = (image_width - kernel_width) // self.stride + 1

        # Initialize the output image
        output = np.zeros((output_height, output_width))

        # Perform convolution
        for i in range(0, output_height):
            for j in range(0, output_width):
                # Get the current region of the image
                image_region = image_padded[i * self.stride:i * self.stride + kernel_height,
                                            j * self.stride:j * self.stride + kernel_width]

                # Perform element-wise multiplication and summation
                output[i, j] = np.sum(image_region * kernel)

        return output

# The execute function
def execute(image, kernel, padding, stride, dilation):
    # Instantiate the Conv2D object with given padding, stride, and dilation
    conv2d = Convolution()
    
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

