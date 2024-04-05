import numpy as np


class Convolution:
    """
    This class implements a basic convolution operation with padding, stride,
    and dilation functionalities.

    Attributes:
        padding (tuple[int, int]): Padding applied to the image borders 
                                  (default: (0, 0)).
        stride (tuple[int, int]): Stride for iterating through the image 
                                  (default: (1, 1)).
        dilation (tuple[int, int]): Dilation rate for the kernel 
                                   (default: (1, 1)).
    """

    def __init__(self, padding=(0, 0), stride=(1, 1), dilation=(1, 1)):
        """
        Initializes the convolution object with padding, stride, and dilation values.

        Args:
            padding (tuple[int, int], optional): Padding applied to the image 
                                                 borders (default: (0, 0)).
            stride (tuple[int, int], optional): Stride for iterating through the 
                                                 image (default: (1, 1)).
            dilation (tuple[int, int], optional): Dilation rate for the kernel 
                                                   (default: (1, 1)).
        """
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def _pad_image(self, image: np.ndarray) -> np.ndarray:
        """
        Pads the image with zeros based on the specified padding values.

        Args:
            image (np.ndarray): The input image to be padded.

        Returns:
            np.ndarray: The padded image.
        """
        pad_top, pad_bottom = self.padding
        pad_left, pad_right = self.padding
        padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                             mode='constant', constant_values=0)
        return padded_image

    def _get_output_shape(self, image_shape: tuple[int, int], kernel_shape: tuple[int, int]) -> tuple[int, int]:
        """
        Calculates the output shape of the convolution operation.

        Args:
            image_shape (tuple[int, int]): Shape of the input image (height, width).
            kernel_shape (tuple[int, int]): Shape of the kernel (height, width).

        Returns:
            tuple[int, int]: Shape of the output feature map (height, width).
        """
        image_h, image_w = image_shape
        kernel_h, kernel_w = kernel_shape
        dil_h, dil_w = self.dilation

        output_h = int(np.ceil((image_h + 2 * self.padding[0] - (dil_h * (kernel_h - 1) + 1)) / self.stride[0]))
        output_w = int(np.ceil((image_w + 2 * self.padding[1] - (dil_w * (kernel_w - 1) + 1)) / self.stride[1]))
        return output_h, output_w

    def run(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Performs the convolution operation on the given image and kernel.

        Args:
            image (np.ndarray): The input image.
            kernel (np.ndarray): The convolution kernel.

        Returns:
            np.ndarray: The convolved feature map.
        """

        # Handle edge cases for image and kernel sizes
        if image.ndim != 2 or kernel.ndim != 2:
            raise ValueError("Input image and kernel must be 2D arrays.")
        if image.shape[0] < kernel.shape[0] or image.shape[1] < kernel.shape[1]:
            raise ValueError("Image size must be greater than or equal to kernel size.")

        # Pad the image if necessary
        padded_image = self._pad_image(image)

        # Get the output shape
        output_h, output_w = self._get_output_shape(padded_image.shape, kernel.shape)

        # Initialize output feature map
        output = np.zeros((output_h, output_w))

        # Iterate through the padded image with the specified stride and dilation
        for y in range(0, padded_image.shape[0] - kernel.shape[0] + 1, self.stride[0]):
            for x in range(0, padded_image.shape[1] - kernel.shape[1] + 1, self.stride[1]):
                # Extract image patch based on kernel size and dilation
                image_patch = padded_image[y:y + kernel.shape[0] * self.dilation[0]:self.dilation[0],
                                            x:x + kernel.shape[1] * self.dilation[1]:self.dilation[1]]

                # Perform element-wise multiplication and summation (convolution)
                output[y // self.stride[0], x // self.stride[1]] = np.sum(image_patch * kernel)

        return output
