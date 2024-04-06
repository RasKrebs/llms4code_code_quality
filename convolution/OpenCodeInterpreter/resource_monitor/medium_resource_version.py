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



from typing import Tuple
import numpy as np

class Convolution:
    def __init__(self, padding=0, stride=1, dilation=1):
        """ Initialize convolution parameters.

        Parameters
        ----------
        padding : int (default: 0)
            Padding size for input image and kernel

        stride : int (default: 1)
            Stride size for moving the kernel across the image.

        dilation : int (default: 1)
            Dilation size, used to increase the receptive field of the kernel.
        """
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def _add_padding(self, image : np.ndarray, padding: int) -> np.ndarray:
        """ Add padding to the input image.

        Parameters
        ----------
        image : numpy array
            Input image as a numpy array

        padding : int
            Padding size for input image and kernel
        """
        return np.pad(image, (padding, padding))

    def _convolve(self, patch: np.ndarray, kernel: np.ndarray) -> float:
        """ Convolves the given patch with the given kernel."""

        return np.sum(patch * kernel)

    def run(self, image : np.ndarray, kernel: np.ndarray):
        """ Run convolution operation on an image with a specific kernel.

        Parameters
        ----------
        image : numpy array
            Input image as a numpy array

        kernel : numpy array
            Kernel to be used for the convolution operation.
        """
        padded_image = self._add_padding(image, self.padding)

        output_height = int((padded_image.shape[0] - kernel.shape[0]) / self.stride + 1)
        output_width = int((padded_image.shape[1] - kernel.shape[1]) / self.stride + 1)

        result = np.zeros((output_height, output_width))

        for h in range(0, output_height):
            for w in range(0, output_width):
                vert_start = self.stride * h
                vert_end = vert_start + kernel.shape[0]

                horiz_start = self.stride * w
                horiz_end = horiz_start + kernel.shape[1]

                patch = padded_image[vert_start:vert_end, horiz_start:horiz_end]

                result[h, w] = self._convolve(patch, kernel)

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

