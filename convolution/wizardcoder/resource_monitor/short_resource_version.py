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
from scipy import signal

def custom_convolution(image, kernel, padding=0, stride=1, dilation=1):
    # Zero Padding
    pad_width = ((padding, padding), (padding, padding))
    image_padded = np.pad(image, pad_width)

    # Applying Dilation
    if dilation > 1:
        image_shape = image_padded.shape
        new_dims = tuple([dilation*i for i in image_shape])

        image_dilated = np.zeros(new_dims)
        h, w = image_shape
        h_new, w_new = new_dims

        # Copy original pixels into new positions
        for x in range(h):
            for y in range(w):
                image_dilated[x*dilation][y*dilation] = image_padded[x][y]

        image_padded = image_dilated

    # Convolution with Stride
    output_shape = (image_padded.shape[0] - kernel.shape[0]) // stride + 1, \
                   (image_padded.shape[1] - kernel.shape[1]) // stride + 1

    result = np.zeros(output_shape)

    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            x, y = i*stride, j*stride
            receptive_field = image_padded[x:x+kernel.shape[0],y:y+kernel.shape[1]]

            result[i][j] = np.sum(np.multiply(receptive_field, kernel))

    return result

# The execute function
def execute(image, kernel, padding, stride, dilation):
    # Instantiate the Conv2D object with given padding, stride, and dilation
    conv2d = custom_convolution(image, kernel, padding, stride, dilation)
    
    return conv2d



if __name__ == "__main__":
    # Start the resource monitoring in a separate thread
    global monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=resource_monitor)
    monitor_thread.start()

    # Execute the Huffman coding process
    con = execute(image, kernel, padding[0], stride[0], dilation[0])


    # Stop the monitoring
    monitoring = False
    monitor_thread.join()

    print(max_resources_usage['cpu']), print(max_resources_usage['memory'] / (1024**2))

