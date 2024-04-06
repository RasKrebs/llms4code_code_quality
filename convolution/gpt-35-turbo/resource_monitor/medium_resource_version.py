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
    def __init__(self, padding: int = 0, stride: int = 1, dilation: int = 1):
        """
        Initialize the Convolution object with padding, stride, and dilation parameters.

        Parameters:
        padding (int): amount of padding to add to the image
        stride (int): stride of the convolution operation
        dilation (int): dilation factor for kernel
        """
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def _pad_image(self, image: np.array, kernel_size: int) -> np.array:
        """
        Pad the image based on the specified padding amount.

        Parameters:
        image (np.array): input image
        kernel_size (int): size of the kernel

        Returns:
        np.array: padded image
        """
        padded_image = np.pad(image, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        return padded_image

    def _apply_kernel(self, image_patch: np.array, kernel: np.array) -> float:
        """
        Apply the kernel to a patch of the image.

        Parameters:
        image_patch (np.array): patch of the image
        kernel (np.array): kernel

        Returns:
        float: result of the convolution
        """
        return np.sum(image_patch * kernel)

    def _get_output_size(self, image: np.array, kernel: np.array) -> tuple:
        """
        Calculate the output size of the convolution operation.

        Parameters:
        image (np.array): input image
        kernel (np.array): kernel

        Returns:
        tuple: output size of the convolution
        """
        output_height = ((image.shape[0] + 2 * self.padding - kernel.shape[0]) // self.stride) + 1
        output_width = ((image.shape[1] + 2 * self.padding - kernel.shape[1]) // self.stride) + 1
        return (output_height, output_width)

    def run(self, image: np.array, kernel: np.array) -> np.array:
        """
        Perform convolution operation on the input image using the specified kernel.

        Parameters:
        image (np.array): input image
        kernel (np.array): kernel

        Returns:
        np.array: convolved image
        """
        # Pad the image
        padded_image = self._pad_image(image, kernel.shape[0])

        # Get output size
        output_size = self._get_output_size(padded_image, kernel)

        # Initialize empty output image
        convolved_image = np.zeros(output_size)

        # Perform convolution
        for i in range(0, padded_image.shape[0] - kernel.shape[0] + 1, self.stride):
            for j in range(0, padded_image.shape[1] - kernel.shape[1] + 1, self.stride):
                patch = padded_image[i: i + kernel.shape[0], j: j + kernel.shape[1]]
                convolved_image[i // self.stride, j // self.stride] = self._apply_kernel(patch, kernel)

        return convolved_image

# The execute function
def execute(image, kernel, padding, stride, dilation):
    # Instantiate the Conv2D object with given padding, stride, and dilation
    conv2d = Convolution(padding[0], stride[0], dilation[0])
    
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

