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

class Conv2D:
    """
    Class to perform 2-dimensional convolution from scratch.

    Parameters
    ----------
    padding : tuple of ints, optional (default=(0, 0))
        Zero-padding added to both sides of the input image.

    stride : int, optional (default=1)
        Stride of the convolution kernel.

    dilation: int, optional (default=1)
        Spacing between kernel elements.

    Methods
    -------
    run(image, kernel):
       Returns the convolved image after applying the convolution operator.
    """

    def __init__(self, padding: Tuple[int, int] = (0, 0), stride: int = 1, dilation: int = 1):
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def _handle_image_edge(self, image: np.ndarray) -> np.ndarray:
        """
        Handles the edge case where an input image is smaller than a kernel.

        Parameters
        ----------
        image : array-like
            Input image to be handled.

        Returns
        -------
        output_image : array-like
           The padded input image after handling the edge case.
        """
        paddings = ((0, max(self.padding[0] - image.shape[0], 0)), (0, max(self.padding[1] - image.shape[1],
0)))
        output_image = np.pad(image, pad_width=paddings)

        return output_image

    def _handle_channels(self, image: np.ndarray, kernel: np.ndarray) -> Tuple[np.ndarray]:
        """
        Handles the edge case where an input image has a different number of channels than the kernel.

        Parameters
        ----------
        image : array-like
            Input image to be handled.

        kernel : array-like
            Convolution kernel to be handled.

        Returns
        -------
        output_image: array-like
           The input image after handling the edge case.

        output_kernel: array-like
           The convolution kernel after handling the edge case.
        """
        if image.shape[0] < kernel.shape[0]:
            paddings = ((0, kernel.shape[0] - image.shape[0]), (0, 0))
            output_image = np.pad(image, pad_width=paddings)
        else:
            output_image = image

        if image.shape[1] < kernel.shape[1]:
            paddings = ((0, 0), (0, kernel.shape[1] - image.shape[1]))
            output_image = np.pad(image, pad_width=paddings)
        else:
            output_image = image

        if kernel.shape[0] < image.shape[0]:
            paddings = ((0, image.shape[0] - kernel.shape[0]), (0, 0))
            output_kernel = np.pad(kernel, pad_width=paddings)

        if kernel.shape[1] < image.shape[1]:
            paddings = ((0, 0), (0, image.shape[1] - kernel.shape[1]))
            output_kernel = np.pad(kernel, pad_width=paddings)

        return output_image, output_kernel

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray):
        """
        Performs 2-dimensional convolution on an input image and a kernel.

        Parameters
        ----------
        image : array-like
            Input image to be convolved.

        kernel : array-like
           Convolution kernel to be applied on the input image.

        Returns
        -------
        output_image: array-like
           The convolved image after applying the convolution operator.
         """
        # Add padding to image
        padded_img = np.pad(image, pad_width=self.padding)

        # Calculate convolution output size
        h_out = (padded_img.shape[0] - kernel.shape[0]) // self.stride + 1
        w_out = (padded_img.shape[1] - kernel.shape[1]) // self.stride + 1

        # Initialize convolution output image
        conv_image = np.zeros((h_out, w_out))

        for i in range(conv_image.shape[0]):
            y_img = i * self.stride

            for j in range(conv_image.shape[1]):
                x_img = j * self.stride

                conv_image[i, j] = np.sum(padded_img[y_img: y_img + kernel.shape[0], x_img:x_img + kernel.shape[1]] * kernel)

        return conv_image

    def run(self, image: np.ndarray, kernel: np.ndarray):
        """
        Returns the convolved image after applying the convolution operator.

        Parameters
        ----------
        image : array-like
            Input image to be convolved.

        kernel : array-like
           Convolution kernel to be applied on the input image.

        Returns
        -------
        output_image: array-like
           The convolved image after applying the convolution operator.
         """
        # Handle edge case where an input image is smaller than a kernel
        if (image.shape[0] < kernel.shape[0]) or (image.shape[1] < kernel.shape[1]):
            image = self._handle_image_edge(image)

        # Handle the edge case where an input image has a different number of channels than the kernel
        if (image.shape[0] != kernel.shape[0]) or (image.shape[1] != kernel.shape[1]):
            image, kernel = self._handle_channels(image, kernel)

        return self._convolve2d(image, kernel)

# The execute function
def execute(image, kernel, padding, stride, dilation):
    # Instantiate the Conv2D object with given padding, stride, and dilation
    conv2d = Conv2D(padding[0], stride[0], dilation[0])
    
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

