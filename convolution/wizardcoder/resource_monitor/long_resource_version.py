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
from typing import Union, Tuple

class Convolution2D:
    """A 2-Dimensional convolution operation."""

    def __init__(self, padding: Union[int, Tuple] = 0, stride: Union[int, Tuple] = 1):
        self.padding = self._check_tuple(padding, 'padding')
        self.stride = self._check_tuple(stride, 'stride')

    def _check_tuple(self, value, name):
        """Check if the input is a tuple or integer and convert it to a tuple."""
        if isinstance(value, int):
            return (value, value)
        elif isinstance(value, tuple):
            assert len(value) == 2, f"{name} should be a 2-tuple."
            assert all(isinstance(i, int) for i in value), f"{name} values should be integers."
            return value
        else:
            raise ValueError(f"Invalid {name}. It should be either an integer or a 2-tuple of integers.")

    def run(self, image: np.ndarray, kernel: Union[np.ndarray, list]):
        """Apply the convolution operation on an image and a kernel."""
        # Convert the list to numpy array if necessary
        if isinstance(kernel, list):
            kernel = np.array(kernel)

        # Check the kernel size and shape
        assert len(kernel.shape) == 2, "Kernel should be a 2D matrix."
        assert kernel.shape[0] == kernel.shape[1], "Kernel should be square."

        pad_width = [(p, p) for p in self.padding]
        image_pad = np.pad(image, pad_width, mode='constant')  # apply padding

        # Perform the convolution
        output_height = (image_pad.shape[0]-kernel.shape[0]) // self.stride[0] + 1
        output_width = (image_pad.shape[1]-kernel.shape[1]) // self.stride[1] + 1

        result = np.zeros((output_height, output_width))  # convolution output

        for i in range(0, output_height):  # loop over rows of the image
            for j in range(0, output_width):  # loop over columns of the image
                h_start = i * self.stride[0]
                h_end = h_start + kernel.shape[0]
                w_start = j * self.stride[1]
                w_end = w_start + kernel.shape[1]

                result[i, j] = np.sum(kernel * image_pad[h_start: h_end, w_start: w_end])  # multiply and sum over the neighborhood

        return result

# The execute function
def execute(image, kernel, padding, stride, dilation):
    # Instantiate the Conv2D object with given padding, stride, and dilation
    conv2d = Convolution2D(padding, stride)
    
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

