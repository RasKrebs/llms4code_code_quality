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

class Convolution2D:
    def __init__(self, padding: int, stride: int, dilation: int):
        """
        Initialize the Convolution2D class.

        :param padding: Padding around the image
        :param stride: Stride for the convolution
        :param dilation: Dilation for the convolution
        """
        self.padding = self._handle_input(padding)
        self.stride = self._handle_input(stride)
        self.dilation = self._handle_input(dilation)

    @staticmethod
    def _handle_input(value):
        """
        Handle the input parameters.

        :param value: The input value
        :return: A tuple representing the value
        """
        if isinstance(value, int):
            return (value, value)
        elif isinstance(value, tuple) and len(value) == 2:
            return value
        else:
            raise ValueError("Input should be an integer or a tuple of two integers.")

    def run(self, image: np.ndarray, kernel: np.ndarray):
        """
        Run the convolution operation.

        :param image: The input image
        :param kernel: The convolution kernel
        :return: The convolved image
        """
        if isinstance(kernel, list):
            kernel = np.array(kernel)

        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Kernel should be a square matrix.")

        if kernel.shape[0] > image.shape[0] or kernel.shape[1] > image.shape[1]:
            raise ValueError("Kernel should not be larger than the image.")

        # Add padding to the image
        padded_image = np.pad(image, self.padding, mode='constant')

        # Calculate the size of the output image
        output_size = ((padded_image.shape[0] - kernel.shape[0]) // self.stride[0] + 1,
                       (padded_image.shape[1] - kernel.shape[1]) // self.stride[1] + 1)

        # Initialize the output image
        output_image = np.zeros(output_size)

        # Perform the convolution operation
        for i in range(0, padded_image.shape[0] - kernel.shape[0] + 1, self.stride[0]):
            for j in range(0, padded_image.shape[1] - kernel.shape[1] + 1, self.stride[1]):
                output_image[i // self.stride[0], j // self.stride[1]] = np.sum(
                    padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel)

        return output_image

# The execute function
def execute(image, kernel, padding, stride, dilation):
    # Instantiate the Conv2D object with given padding, stride, and dilation
    conv2d = Convolution2D(padding, stride, dilation)
    
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

