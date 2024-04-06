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

class Convolution:
    def __init__(self, padding: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], dilation:
Union[int, Tuple[int, int]]):
        self.padding = (padding, padding) if isinstance(padding, int) else (padding[0], padding[1])
        self.stride = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else (dilation[0], dilation[1])

    def run(self, image: np.ndarray, kernel: Union[list, np.ndarray]):
        # Ensure the image and the kernel are numpy arrays
        image = np.array(image) if not isinstance(image, np.ndarray) else image
        kernel = np.array(kernel) if not isinstance(kernel, np.ndarray) else kernel

        # Add padding to the image
        image_padded = self._pad_image(image)


        # Apply the convolution
        result = self._convolution(image_padded, kernel)

        return result

    def _pad_image(self, image: np.ndarray):
        h, w = image.shape
        padded_image = np.zeros((h + 2 * self.padding[0], w + 2 * self.padding[1]))
        padded_image[self.padding[0]:self.padding[0] + h, self.padding[1]:self.padding[1] + w] = image[:, :]

        return padded_image

    def _convolution(self, image: np.ndarray, kernel: np.ndarray):
        h, w = image.shape
        kh, kw = kernel.shape

        # Calculate the output size
        out_height = (h - kh) // self.stride[0] + 1
        out_width = (w - kw) // self.stride[1] + 1

        # Initialize the output array
        result = np.zeros((out_height, out_width))

        for i in range(0, h - kh + 1, self.stride[0]):
            for j in range(0, w - kw + 1, self.stride[1]):
                output = np.sum(image[i:i+kh, j:j+kw] * kernel[:, :])
                result[i // self.stride[0], j // self.stride[1]] = output

        return result

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

