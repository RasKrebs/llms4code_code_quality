import numpy as np

class Convolution:
    def __init__(self, padding: int, stride: int, dilation: int):
        """
        Initialize Convolution class with padding, stride, and dilation parameters.
        
        Parameters:
        padding (int or tuple): Padding value for convolution operation
        stride (int): Stride value for convolution operation
        dilation (int): Dilation value for convolution operation
        """
        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple) and len(padding) == 2:
            self.padding = padding
        else:
            raise ValueError("Padding must be an integer or a tuple of length 2")
        
        if not isinstance(stride, int):
            raise ValueError("Stride must be an integer")
        self.stride = stride
        
        if not isinstance(dilation, int):
            raise ValueError("Dilation must be an integer")
        self.dilation = dilation
        
    def _pad_image(self, image: np.ndarray, padding: tuple) -> np.ndarray:
        """
        Pad the input image based on the padding value
        
        Parameters:
        image (np.ndarray): Input image for convolution
        padding (tuple): Padding values for height and width
        
        Returns:
        np.ndarray: Padded image
        """
        padded_image = np.pad(image, ((padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
        return padded_image
    
    def _check_kernel(self, kernel: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Check if the kernel is valid for the image
        
        Parameters:
        kernel (np.ndarray): Kernel matrix for convolution
        image (np.ndarray): Input image
        
        Returns:
        np.ndarray: Valid kernel matrix
        """
        if kernel.ndim == 1:
            kernel = np.array(kernel).reshape((int(np.sqrt(len(kernel))), int(np.sqrt(len(kernel)))))
        
        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Kernel must be a square matrix")
        
        if kernel.shape[0] > image.shape[0] or kernel.shape[1] > image.shape[1]:
            raise ValueError("Kernel size must be smaller than the image size")
        
        return kernel
    
    def run(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Perform convolution on the input image with the given kernel
        
        Parameters:
        image (np.ndarray): Input image for convolution
        kernel (np.ndarray): Kernel matrix for convolution
        
        Returns:
        np.ndarray: Convolved image
        """
        padded_image = self._pad_image(image, self.padding)
        kernel = self._check_kernel(kernel, image)
        
        convolved_image = np.zeros((image.shape[0], image.shape[1]))
        
        for i in range(0, image.shape[0] - kernel.shape[0] + 1, self.stride):
            for j in range(0, image.shape[1] - kernel.shape[1] + 1, self.stride):
                receptive_field = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
                convolved_value = np.sum(np.multiply(receptive_field, kernel))
                convolved_image[int(i/self.stride),int(j/self.stride)] = convolved_value
        
        return convolved_image
