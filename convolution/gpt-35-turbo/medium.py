import numpy as np

class Convolution:
    def __init__(self, padding: int = 0, stride: int = 1, dilation: int = 1):
        """
        Initialize the Convolution object with padding, stride, and dilation parameters.

        Parameters:
        padding (int): amount of padding to add to the image
        stride (int): stride of the convolution operation
        dilation (int): dilation factor for kernel
        """
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def _pad_image(self, image: np.array, kernel_size: int) -> np.array:
        """
        Pad the image based on the specified padding amount.

        Parameters:
        image (np.array): input image
        kernel_size (int): size of the kernel

        Returns:
        np.array: padded image
        """
        padded_image = np.pad(image, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        return padded_image

    def _apply_kernel(self, image_patch: np.array, kernel: np.array) -> float:
        """
        Apply the kernel to a patch of the image.

        Parameters:
        image_patch (np.array): patch of the image
        kernel (np.array): kernel

        Returns:
        float: result of the convolution
        """
        return np.sum(image_patch * kernel)

    def _get_output_size(self, image: np.array, kernel: np.array) -> tuple:
        """
        Calculate the output size of the convolution operation.

        Parameters:
        image (np.array): input image
        kernel (np.array): kernel

        Returns:
        tuple: output size of the convolution
        """
        output_height = ((image.shape[0] + 2 * self.padding - kernel.shape[0]) // self.stride) + 1
        output_width = ((image.shape[1] + 2 * self.padding - kernel.shape[1]) // self.stride) + 1
        return (output_height, output_width)

    def run(self, image: np.array, kernel: np.array) -> np.array:
        """
        Perform convolution operation on the input image using the specified kernel.

        Parameters:
        image (np.array): input image
        kernel (np.array): kernel

        Returns:
        np.array: convolved image
        """
        # Pad the image
        padded_image = self._pad_image(image, kernel.shape[0])

        # Get output size
        output_size = self._get_output_size(padded_image, kernel)

        # Initialize empty output image
        convolved_image = np.zeros(output_size)

        # Perform convolution
        for i in range(0, padded_image.shape[0] - kernel.shape[0] + 1, self.stride):
            for j in range(0, padded_image.shape[1] - kernel.shape[1] + 1, self.stride):
                patch = padded_image[i: i + kernel.shape[0], j: j + kernel.shape[1]]
                convolved_image[i // self.stride, j // self.stride] = self._apply_kernel(patch, kernel)

        return convolved_image