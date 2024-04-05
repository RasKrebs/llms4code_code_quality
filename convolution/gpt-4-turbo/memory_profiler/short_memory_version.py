from memory_profiler import profile
        import os
        import psutil
        
        # Get the current process ID
        pid = os.getpid()

        # Create a psutil Process object for the current process
        process = psutil.Process(pid)

        # Get the number of logical CPUs in the system
        num_cores = psutil.cpu_count(logical=True)import numpy as np

@profile
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
    
    return output_matrix<write execution script for convolution>
