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



def convolve2d(image, kernel, padding=0, stride=1):
    # Input image dimensions
    i_h, i_w = len(image), len(image[0])

    # Kernel dimensions
    k_h, k_w = len(kernel), len(kernel[0])

    # Output image dimensions
    o_h = (i_h - k_h + 2*padding) // stride + 1
    o_w = (i_w - k_w + 2*padding) // stride + 1

    # Create an empty output image
    out_image = [[0 for _ in range(o_w)] for _ in range(o_h)]

    # Add padding to input image
    if padding:
        padded_image = [[0 for _ in range(i_w + 2*padding)] for _ in range(i_h + 2*padding)]

        for i in range(len(image)):
            for j in range(len(image[0])):
                padded_image[i+padding][j+padding] = image[i][j]

    else:
        padded_image = image

    # Perform convolution operation
    for i in range(o_h):
        for j in range(o_w):
            for m in range(k_h):
                for n in range(k_w):
                    out_image[i][j] += padded_image[i*stride+m][j*stride+n] * kernel[m][n]

    return out_image

# The execute function
def execute(image, kernel, padding, stride, dilation):
    # Instantiate the Conv2D object with given padding, stride, and dilation
    conv2d = convolve2d(image, kernel, padding[0], stride[0])
    
    # Run the convolution process
    #convolved_image = conv2d.run(image, kernel)
    
    return conv2d



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

