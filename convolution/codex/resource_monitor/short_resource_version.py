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

def conv2d(image, kernel, padding=0, stride=1, dilation=1):
    # Dimensions of input image and kernel
    (image_height, image_width) = image.shape
    (kernel_height, kernel_width) = kernel.shape

    # Calculate the dimensions of the output
    output_height = (image_height + 2*padding - dilation*(kernel_height-1) - 1)//stride + 1
    output_width = (image_width + 2*padding - dilation*(kernel_width-1) - 1)//stride + 1

    # Initialize the output with zeros
    output = np.zeros((output_height, output_width))

    # Pad the image
    image_padded = np.pad(image, padding)

    # Apply the kernel to the image
    for x in range(output_width):
        for y in range(output_height):
            # Extract a patch of the image
            patch = image_padded[y*stride:y*stride+kernel_height, x*stride:x*stride+kernel_width]
            # Apply dilation
            if dilation > 1:
                patch = patch[::dilation, ::dilation]
            # Apply the kernel to the patch
            output[y, x] = np.sum(patch * kernel)

    return output

# The execute function
def execute(image, kernel, padding, stride, dilation):
    # Instantiate the Conv2D object with given padding, stride, and dilation
    con = conv2d(image, kernel)

    return con



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

