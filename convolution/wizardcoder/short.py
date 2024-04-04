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