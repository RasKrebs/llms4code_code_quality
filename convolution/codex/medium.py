import numpy as np

class Convolution:
    """
    A class used to perform convolution on an image

    ...

    Attributes
    ----------
    padding : int
        the padding size (default is 0)
    stride : int
        the stride size (default is 1)
    dilation : int
        the dilation size (default is 1)

    Methods
    -------
    run(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        Performs convolution on the image using the kernel
    """

    def __init__(self, padding: int = 0, stride: int = 1, dilation: int = 1):
        """
        Parameters
        ----------
        padding : int, optional
            The padding size (default is 0)
        stride : int, optional
            The stride size (default is 1)
        dilation : int, optional
            The dilation size (default is 1)
        """
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def run(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Performs convolution on the image using the kernel

        Parameters
        ----------
        image : np.ndarray
            The input image
        kernel : np.ndarray
            The kernel to convolve the image with

        Returns
        -------
        np.ndarray
            The convolved image
        """
        # Add padding to the image
        if self.padding > 0:
            image = np.pad(image, self.padding)

        # Flip the kernel
        kernel = np.flipud(np.fliplr(kernel))

        # Calculate the dimensions of the output image
        x_output = int(((image.shape[0] - kernel.shape[0] + 2 * self.padding) / self.stride) + 1)
        y_output = int(((image.shape[1] - kernel.shape[1] + 2 * self.padding) / self.stride) + 1)
        output = np.zeros((x_output, y_output))

        # Perform convolution
        for x in range(image.shape[1]):
            if x > image.shape[1] - kernel.shape[1]:
                break
            if x % self.stride == 0:
                for y in range(image.shape[0]):
                    if y > image.shape[0] - kernel.shape[0]:
                        break
                    try:
                        if y % self.stride == 0:
                            output[x, y] = (kernel * image[x: x + kernel.shape[1], y: y + kernel.shape[0]]).sum()
                    except:
                        break

        return output