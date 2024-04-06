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
    def __init__(self, padding: Union[int, Tuple[int, int]] = 0, stride: Union[int, Tuple[int, int]] = 1, dilation: Union[int, Tuple[int, int]] = 1):
        """
        Initialize the convolution operation with padding, stride and dilation.

        Parameters:
        padding (Union[int, Tuple[int, int]]): Padding to be applied to the image. If int, same padding is applied to all sides.
        stride (Union[int, Tuple[int, int]]): Stride of the convolution operation. If int, same stride is applied to both dimensions.
        dilation (Union[int, Tuple[int, int]]): Dilation of the convolution operation. If int, same dilation is applied to both dimensions.
        """
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

    def run(self, image: np.ndarray, kernel: Union[np.ndarray, list]):
        """
        Run the convolution operation on the given image with the given kernel.

        Parameters:
        image (np.ndarray): The image to be convolved.
        kernel (Union[np.ndarray, list]): The kernel to be used in the convolution.

        Returns:
        np.ndarray: The convolved image.
        """
        # Convert kernel to numpy array if it's a list
        if isinstance(kernel, list):
            kernel = np.array(kernel)

        # Check if kernel is a square matrix
        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Kernel must be a square matrix.")

        # Check if kernel is the correct size for the image
        if kernel.shape[0] > image.shape[0] or kernel.shape[1] > image.shape[1]:
            raise ValueError("Kernel must be smaller than the image.")

        # Add padding to the image
        image = np.pad(image, self.padding, mode='constant')

        # Initialize output image
        output_shape = ((image.shape[0] - kernel.shape[0]) // self.stride[0] + 1,
                        (image.shape[1] - kernel.shape[1]) // self.stride[1] + 1)
        output = np.zeros(output_shape)

        # Perform convolution
        for i in range(0, output.shape[0], self.stride[0]):
            for j in range(0, output.shape[1], self.stride[1]):
                output[i, j] = np.sum(image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)

        return output

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

