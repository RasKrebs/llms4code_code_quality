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

class ConvolutionLayer:
    def __init__(self, filter_size, stride=1, padding='valid', dilation_rate=1):
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.dilation_rate = dilation_rate

    def create_filters(self, channel_num):
        return np.random.randn(self.filter_size, self.filter_size, channel_num) / 9

    def zero_padding(self, image):
        if self.padding == 'valid':
            return image
        else:
            pad_width = ((0,0), (self.filter_size-1,self.filter_size-1),
                         (self.filter_size-1,self.filter_size-1))
            return np.pad(image, pad_width=pad_width)

    def forward(self, image):
        # Adding padding to the input image if required
        padded_image = self.zero_padding(image)

        output_shape = ((padded_image.shape[0] - self.filter_size + 1) // self.stride, (padded_image.shape[1] -
                                                                                        self.filter_size + 1) // self.stride)

        # Create filters if not already created
        if not hasattr(self, 'filters'):
            channel_num = padded_image.shape[-1] if len(padded_image.shape) > 2 else 1
            self.filters = self.create_filters(channel_num)

        output = np.zeros((output_shape[0], output_shape[1]))

        for i in range(0, padded_image.shape[0] - self.filter_size + 1, self.stride):
            for j in range(0, padded_image.shape[1] - self.filter_size + 1, self.stride):
                output[i // self.stride][j // self.stride] = np.sum(padded_image[i: i + self.filter_size, j: j
                                                                                 + self.filter_size] * self.filters)

        return output

# The execute function
def execute(image, kernel, padding, stride, dilation):
    # Instantiate the Conv2D object with given padding, stride, and dilation
    conv2d = ConvolutionLayer(filter_size=kernel.shape[0], stride=stride[0], padding='valid', dilation_rate=dilation[0])
    
    # Run the convolution process
    convolved_image = conv2d.forward(image)
    
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

