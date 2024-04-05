import numpy as np

def conv2d(input_image, kernel, padding=0, stride=1, dilation=1):
    # Get dimensions of input image and kernel
    img_height, img_width = input_image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate output dimensions
    out_height = (img_height + 2*padding - dilation*(kernel_height - 1) - 1) // stride + 1
    out_width = (img_width + 2*padding - dilation*(kernel_width - 1) - 1) // stride + 1
    
    # Create zero-padded input image
    padded_image = np.pad(input_image, padding)
    
    # Initialize output image
    output_image = np.zeros((out_height, out_width))
    
    for i in range(0, out_height, stride):
        for j in range(0, out_width, stride):
            # Extract receptive field from input image
            rf = padded_image[i:i+kernel_height, j:j+kernel_width]
            
            # Apply convolution operation
            output_image[i, j]
