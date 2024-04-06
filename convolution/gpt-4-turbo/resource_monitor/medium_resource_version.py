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
        Initializes a Convolution object with specified padding, stride, and dilation.

        Parameters:
        - padding: Number of zero-padding layers around the image. Default is 0.
        - stride: Number of pixels to skip while sliding the filter over the image. Default is 1.
        - dilation: Spacing between kernel elements. Default is 1.
        """
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def _add_padding(self, image: np.ndarray) -> np.ndarray:
        """Adds padding to the image if padding is greater than 0."""
        if self.padding > 0:
            return np.pad(image, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
        return image

    def _calculate_output_dimensions(self, image_shape: tuple, kernel_shape: tuple) -> tuple:
        """
        Calculates the dimensions of the output image after convolution.

        Parameters:
        - image_shape: A tuple representing the shape of the input image.
        - kernel_shape: A tuple representing the shape of the kernel.

        Returns:
        A tuple representing the shape of the output image.
        """
        n_h, n_w = image_shape
        f_h, f_w = kernel_shape

        h = ((n_h + 2 * self.padding - f_h - (f_h - 1) * (self.dilation - 1)) // self.stride) + 1
        w = ((n_w + 2 * self.padding - f_w - (f_w - 1) * (self.dilation - 1)) // self.stride) + 1

        return h, w

    def run(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Performs convolution on the input image using the specified kernel.

        Parameters:
        - image: A 2D numpy array representing the input image.
        - kernel: A 2D numpy array representing the filter/kernel.

        Returns:
        A 2D numpy array representing the convolved image.
        """
        # Ensure kernel and image are 2D arrays
        if image.ndim != 2 or kernel.ndim != 2:
            raise ValueError("Both image and kernel must be 2D arrays.")

        # Add padding to the image
        padded_image = self._add_padding(image)
        
        # Calculate output dimensions
        output_dimensions = self._calculate_output_dimensions(image.shape, kernel.shape)
        
        # Initialize the output image with zeros
        convolved_image = np.zeros(output_dimensions)
        
        # Flip the kernel for the convolution operation (cross-correlation otherwise)
        flipped_kernel = np.flip(kernel)

        # Apply the kernel/filter over the image
        for i in range(convolved_image.shape[0]):
            for j in range(convolved_image.shape[1]):
                # Calculate the region of interest in the image
                start_i = i * self.stride
                end_i = start_i + flipped_kernel.shape[0] * self.dilation
                start_j = j * self.stride
                end_j = start_j + flipped_kernel.shape[1] * self.dilation
                
                # Extract the region of interest
                region = padded_image[start_i:end_i:self.dilation, start_j:end_j:self.dilation]
                
                # Ensure the region and kernel have the same size before convolution
                if region.shape == flipped_kernel.shape:
                    # Element-wise multiplication and sum (convolution operation)
                    convolved_image[i, j] = np.sum(region * flipped_kernel)

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

