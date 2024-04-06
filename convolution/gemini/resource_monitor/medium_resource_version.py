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


class Convolution:
    """
    This class implements a basic convolution operation with padding, stride,
    and dilation functionalities.

    Attributes:
        padding (tuple[int, int]): Padding applied to the image borders 
                                  (default: (0, 0)).
        stride (tuple[int, int]): Stride for iterating through the image 
                                  (default: (1, 1)).
        dilation (tuple[int, int]): Dilation rate for the kernel 
                                   (default: (1, 1)).
    """

    def __init__(self, padding=(0, 0), stride=(1, 1), dilation=(1, 1)):
        """
        Initializes the convolution object with padding, stride, and dilation values.

        Args:
            padding (tuple[int, int], optional): Padding applied to the image 
                                                 borders (default: (0, 0)).
            stride (tuple[int, int], optional): Stride for iterating through the 
                                                 image (default: (1, 1)).
            dilation (tuple[int, int], optional): Dilation rate for the kernel 
                                                   (default: (1, 1)).
        """
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def _pad_image(self, image: np.ndarray) -> np.ndarray:
        """
        Pads the image with zeros based on the specified padding values.

        Args:
            image (np.ndarray): The input image to be padded.

        Returns:
            np.ndarray: The padded image.
        """
        pad_top, pad_bottom = self.padding
        pad_left, pad_right = self.padding
        padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                             mode='constant', constant_values=0)
        return padded_image

    def _get_output_shape(self, image_shape: tuple[int, int], kernel_shape: tuple[int, int]) -> tuple[int, int]:
        """
        Calculates the output shape of the convolution operation.

        Args:
            image_shape (tuple[int, int]): Shape of the input image (height, width).
            kernel_shape (tuple[int, int]): Shape of the kernel (height, width).

        Returns:
            tuple[int, int]: Shape of the output feature map (height, width).
        """
        image_h, image_w = image_shape
        kernel_h, kernel_w = kernel_shape
        dil_h, dil_w = self.dilation

        output_h = int(np.ceil((image_h + 2 * self.padding[0] - (dil_h * (kernel_h - 1) + 1)) / self.stride[0]))
        output_w = int(np.ceil((image_w + 2 * self.padding[1] - (dil_w * (kernel_w - 1) + 1)) / self.stride[1]))
        return output_h, output_w

    def run(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Performs the convolution operation on the given image and kernel.

        Args:
            image (np.ndarray): The input image.
            kernel (np.ndarray): The convolution kernel.

        Returns:
            np.ndarray: The convolved feature map.
        """

        # Handle edge cases for image and kernel sizes
        if image.ndim != 2 or kernel.ndim != 2:
            raise ValueError("Input image and kernel must be 2D arrays.")
        if image.shape[0] < kernel.shape[0] or image.shape[1] < kernel.shape[1]:
            raise ValueError("Image size must be greater than or equal to kernel size.")

        # Pad the image if necessary
        padded_image = self._pad_image(image)

        # Get the output shape
        output_h, output_w = self._get_output_shape(padded_image.shape, kernel.shape)

        # Initialize output feature map
        output = np.zeros((output_h, output_w))

        # Iterate through the padded image with the specified stride and dilation
        for y in range(0, padded_image.shape[0] - kernel.shape[0] + 1, self.stride[0]):
            for x in range(0, padded_image.shape[1] - kernel.shape[1] + 1, self.stride[1]):
                # Extract image patch based on kernel size and dilation
                image_patch = padded_image[y:y + kernel.shape[0] * self.dilation[0]:self.dilation[0],
                                            x:x + kernel.shape[1] * self.dilation[1]:self.dilation[1]]

                # Perform element-wise multiplication and summation (convolution)
                output[y // self.stride[0], x // self.stride[1]] = np.sum(image_patch * kernel)

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

