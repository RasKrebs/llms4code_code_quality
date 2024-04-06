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

def convolve2d(image: np.ndarray, kernel: np.ndarray, pad_width: int = 0, stride: int = 1, dilation: int = 1) -> np.ndarray:
    """Perform a valid 2D convolution operation on the input image with the given kernel."""

    # Pad the image based on the padding width
    if pad_width > 0:
        image = np.pad(image, pad_width=pad_width)

    # Compute the output shape (height and width)
    out_height = (image.shape[0] - kernel.shape[0]) // stride + 1
    out_width = (image.shape[1] - kernel.shape[1]) // stride + 1

    # Initialize the output array with zeros
    out_array = np.zeros((out_height, out_width))

    # Perform convolution operation
    for i in range(0, image.shape[0] - kernel.shape[0] + 1, stride):
        for j in range(0, image.shape[1] - kernel.shape[1] + 1, stride):
            window = image[i:i+kernel.shape[0]:dilation, j:j+kernel.shape[1]:dilation]
            out_array[i//stride, j//stride] = np.sum(window * kernel)

    return out_array

# The execute function
def execute(image, kernel, padding, stride, dilation):
    # Instantiate the Conv2D object with given padding, stride, and dilation
    conv2d = convolve2d(image, kernel)
    
    return conv2d



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

