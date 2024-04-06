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

def conv2d(input_matrix, kernel, padding=0, stride=1, dilation=1):
    # Extracting input and kernel dimensions
    in_height, in_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape
    
    # Adjusting kernel size based on dilation
    adj_kernel_height = kernel_height + (kernel_height - 1) * (dilation - 1)
    adj_kernel_width = kernel_width + (kernel_width - 1) * (dilation - 1)
    
    # Calculating the dimensions of the output matrix
    output_height = (in_height + 2*padding - adj_kernel_height) // stride + 1
    output_width = (in_width + 2*padding - adj_kernel_width) // stride + 1
    
    # Padding
    padded_input = np.pad(input_matrix, padding, mode='constant', constant_values=0)
    
    # Initializing output matrix with zeros
    output_matrix = np.zeros((output_height, output_width))
    
    # Performing the convolution
    for y in range(0, output_height):
        for x in range(0, output_width):
            for i in range(adj_kernel_height):
                for j in range(adj_kernel_width):
                    # Calculating the original kernel indices considering dilation
                    if i % dilation == 0 and j % dilation == 0:
                        # Applying element-wise multiplication and sum to get the convolved value
                        output_matrix[y, x] += padded_input[y*stride+i, x*stride+j] * \
                                               kernel[i//dilation, j//dilation]
    
    return output_matrix

# The execute function
def execute(image, kernel, padding, stride, dilation):
    # Instantiate the Conv2D object with given padding, stride, and dilation
    conv = conv2d(image, kernel, padding[0], stride[0], dilation[0])
    
    return conv



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

