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
