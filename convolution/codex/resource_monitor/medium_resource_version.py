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
    A class used to perform convolution on an image

    ...

    Attributes
    ----------
    padding : int
        the padding size (default is 0)
    stride : int
        the stride size (default is 1)
    dilation : int
        the dilation size (default is 1)

    Methods
    -------
    run(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        Performs convolution on the image using the kernel
    """

    def __init__(self, padding: int = 0, stride: int = 1, dilation: int = 1):
        """
        Parameters
        ----------
        padding : int, optional
            The padding size (default is 0)
        stride : int, optional
            The stride size (default is 1)
        dilation : int, optional
            The dilation size (default is 1)
        """
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def run(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Performs convolution on the image using the kernel

        Parameters
        ----------
        image : np.ndarray
            The input image
        kernel : np.ndarray
            The kernel to convolve the image with

        Returns
        -------
        np.ndarray
            The convolved image
        """
        # Add padding to the image
        if self.padding > 0:
            image = np.pad(image, self.padding)

        # Flip the kernel
        kernel = np.flipud(np.fliplr(kernel))

        # Calculate the dimensions of the output image
        x_output = int(((image.shape[0] - kernel.shape[0] + 2 * self.padding) / self.stride) + 1)
        y_output = int(((image.shape[1] - kernel.shape[1] + 2 * self.padding) / self.stride) + 1)
        output = np.zeros((x_output, y_output))

        # Perform convolution
        for x in range(image.shape[1]):
            if x > image.shape[1] - kernel.shape[1]:
                break
            if x % self.stride == 0:
                for y in range(image.shape[0]):
                    if y > image.shape[0] - kernel.shape[0]:
                        break
                    try:
                        if y % self.stride == 0:
                            output[x, y] = (kernel * image[x: x + kernel.shape[1], y: y + kernel.shape[0]]).sum()
                    except:
                        break

        return output

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

