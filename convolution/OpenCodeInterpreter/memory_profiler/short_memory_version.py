from memory_profiler import profile
        import os
        import psutil
        
        # Get the current process ID
        pid = os.getpid()

        # Create a psutil Process object for the current process
        process = psutil.Process(pid)

        # Get the number of logical CPUs in the system
        num_cores = psutil.cpu_count(logical=True)@profile
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

    return out_image<write execution script for convolution>
