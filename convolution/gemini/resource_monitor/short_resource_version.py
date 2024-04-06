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


def convolve(image, kernel, stride=1, padding="same", dilation=1):
  """
  Performs convolution on an image with a kernel.

  Args:
      image: A numpy array representing the input image.
      kernel: A numpy array representing the filter kernel.
      stride: The stride value for moving the kernel (default: 1).
      padding: The padding mode ("same" or "valid") (default: "same").
      dilation: The dilation rate for applying the kernel (default: 1).

  Returns:
      A numpy array representing the convolved feature map.
  """

  # Get image and kernel dimensions
  image_h, image_w = image.shape[:2]
  kernel_h, kernel_w = kernel.shape

  # Validate input shapes
  if (image_h + 2 * ((padding == "same") and (dilation > 1)) - kernel_h) / stride < 0 or \
      (image_w + 2 * ((padding == "same") and (dilation > 1)) - kernel_w) / stride < 0:
    raise ValueError("Invalid padding or image/kernel size for given stride and dilation")

  # Calculate output dimensions based on padding mode
  if padding == "same":
    out_h = int(ceil((image_h + dilation * (kernel_h - 1)) / stride))
    out_w = int(ceil((image_w + dilation * (kernel_w - 1)) / stride))
  elif padding == "valid":
    out_h = int(max((image_h - kernel_h + 1) / stride, 0))
    out_w = int(max((image_w - kernel_w + 1) / stride, 0))
  else:
    raise ValueError("Invalid padding mode. Choose 'same' or 'valid'.")

  # Handle padding for "same" mode
  if padding == "same":
    pad_top = int((out_h - 1) * stride - image_h + kernel_h) // 2
    pad_bottom = int((out_h - 1) * stride - image_h + kernel_h) - pad_top
    pad_left = int((out_w - 1) * stride - image_w + kernel_w) // 2
    pad_right = int((out_w - 1) * stride - image_w + kernel_w) - pad_left
    image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

  # Initialize output feature map
  output = np.zeros((out_h, out_w))

  # Perform convolution loop
  for y in range(0, out_h):
    for x in range(0, out_w):
      # Extract image patch for current position
      image_patch = image[y * stride:y * stride + kernel_h, x * stride:x * stride + kernel_w]

      # Apply dilation to kernel
      dilated_kernel = kernel[::dilation, ::dilation]

      # Reshape dilated_kernel to match the shape of image_patch
      dilated_kernel = dilated_kernel[:image_patch.shape[0], :image_patch.shape[1]]

      # Element-wise multiplication and summation
      output[y, x] = np.sum(image_patch * dilated_kernel)

  return output

def ceil(x):
  """
  Custom ceiling function for compatibility.
  """
  return int(x - 0.1) if x % 1 > 0 else int(x)

# The execute function
def execute(image, kernel, padding, stride, dilation):
    # Instantiate the Conv2D object with given padding, stride, and dilation
    conv2d = convolve(image, kernel)
    
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

