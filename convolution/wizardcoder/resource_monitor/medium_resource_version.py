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
from typing import Tuple, Union

class Convolution2D:
    """A 2-dimensional convolution operation."""

    def __init__(self, padding: int = 0, stride: int = 1, dilation: int = 1):
        """Initialize the convolution.

        Args:
            padding (int): Padding size. Default is 0.
            stride (int): Stride length. Default is 1.
            dilation (int): Dilation rate. Default is 1.
        """
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def run(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply the convolution operation to an image using a specific kernel.

        Args:
            image (np.ndarray): The input image. Must be 2D and real-valued.
            kernel (np.ndarray): The filter kernel. Must be 2D, square and real-valued.

        Returns:
            np.ndarray: The convolved image of same size as the input.
        """
        if self.padding != 0:
            image = self._apply_zero_padding(image, self.padding)

        output_shape = self._get_output_shape(image.shape, kernel.shape, self.stride, self.dilation)
        convolved_image = np.zeros(output_shape)

        for i in range(0, image.shape[0] - kernel.shape[0] + 1, self.stride):
            for j in range(0, image.shape[1] - kernel.shape[1] + 1, self.stride):
                convolved_image[i, j] = np.sum(image[i: i + kernel.shape[0], j: j + kernel.shape[1]] * kernel)

        return convolved_image

    def _apply_zero_padding(self, image: np.ndarray, padding: int):
        """Pad the input image with zeroes on all sides."""
        padded_img = np.zeros((image.shape[0] + 2*padding, image.shape[1] + 2*padding))
        padded_img[padding:-padding, padding:-padding] = image
        return padded_img

    def _get_output_shape(self, image_shape: Tuple[int, int], kernel_shape: Tuple[int, int], stride: int, dilation: int) -> Tuple[int, int]:
        """Calculate the output shape of the convolution operation."""
        out_height = (image_shape[0] - dilation * (kernel_shape[0] - 1) - 1) // stride + 1
        out_width = (image_shape[1] - dilation * (kernel_shape[1] - 1) - 1) // stride + 1

        return out_height, out_width

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

