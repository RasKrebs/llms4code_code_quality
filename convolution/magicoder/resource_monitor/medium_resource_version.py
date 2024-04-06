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
    """Convolution class"""

    def __init__(self, padding: int = 0, stride: int = 1, dilation: int = 1):
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def add_padding(self, image: np.ndarray) -> np.ndarray:
        """Add padding to the image"""

        # Check if padding is greater than 0
        if not self.padding:
            return image

        height, width = image.shape[:2]

        # Add padding to all sides of the image
        padded_image = np.pad(image, ((self.padding,), (self.padding,)))

        if len(image.shape) > 2:  # Check if image is color or grayscale
            num_channels = image.shape[2]
            padded_image = np.pad(padded_image, ((0, 0), (0, 0), (self.padding//2, self.padding//2)))

        return padded_image

    def run(self, image: np.ndarray, kernel: Union[np.ndarray, Tuple]) -> np.ndarray:
        """Run convolution on the image with the provided kernel"""
        # Add padding to the image if necessary
        padded_image = self.add_padding(image)

        if isinstance(kernel, tuple):  # If a kernel size is given as a tuple, create a kernel of that size
            kernel_height, kernel_width = kernel
            kernel = np.ones((kernel_height, kernel_width)) / (kernel_height * kernel_width)
        elif not isinstance(kernel, np.ndarray):  # If the provided kernel is not a numpy array or tuple, raise an error
            raise TypeError("The provided kernel must be a numpy array or a tuple representing the size of the kernel")

        output_height = (padded_image.shape[0] - kernel.shape[0]) // self.stride + 1
        output_width = (padded_image.shape[1] - kernel.shape[1]) // self.stride + 1
        if len(padded_image.shape) > 2:
            num_channels = padded_image.shape[2]  # Get number of channels in image


            conv_img = np.zeros((output_height, output_width, num_channels))   # Create an empty array for theoutput image

            for channel in range(num_channels):  # Convolve each color channel individually
                img_channel = padded_image[:,:,channel]

                conv_img[:,:,channel] = np.array([[np.sum(kernel * img_channel[i*self.stride:(i+kernel.shape[0])*self.stride,
                                                                               j*self.stride:(j+kernel.shape[1])*self.stride])
                                                   for j in range(output_width)] for i in range(output_height)])   # Apply convolution on the channel
        else:
            conv_img = np.array([[np.sum(kernel * padded_image[i*self.stride:(i+kernel.shape[0])*self.stride,
                                                            j*self.stride:(j+kernel.shape[1])*self.stride])
                                for j in range(output_width)]for i in range(output_height)])   # Apply convolution on the image if it is grayscale

        return conv_img

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

