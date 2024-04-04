from memory_profiler import profile
        import os
        import psutil
        
        # Get the current process ID
        pid = os.getpid()

        # Create a psutil Process object for the current process
        process = psutil.Process(pid)

        # Get the number of logical CPUs in the system
        num_cores = psutil.cpu_count(logical=True)# This solution is built drawing inspiration from the following resources:
# https://github.com/detkov/Convolution-From-Scratch/blob/main/convolution.py
from typing import List, Tuple, Union
import numpy as np

class Convolution:
    """Convolution object. This object is used to perform convolution operation on the input
    matrix. It  does not update the input matrix, but returns the result of the convolution
    operation.
    """

    @profile
    def __init__(self,
                padding: Union[int, Tuple[int, int]] = (0, 0),
                stride: Union[int, Tuple[int, int]] = (1, 1),
                dilation: Union[int, Tuple[int, int]] = (1, 1)):
        """Initializes Convolution object. This object is used to perform convolution operation
        on the input matrix. It  does not update the input matrix, but returns the result of the
        convolution operation.

        Args:
            stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)`
                stride we move on `r` pixels along rows and on `c` pixels along columns on each
                iteration. Defaults to (1, 1).

            dilation Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)`
                dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along
                columns. Defaults to (1, 1).

            padding (Union[int, Tuple[int, int]], optional): Tuple with number of rows and columns
                to be padded. With the `(r, c)` padding we addding `r` rows to the top and bottom
                and `c` columns to the left and to the right of ¨the matrix. If padding is an
                integer, then `r = c = padding`.

        Raises:
            ValueError: If stride or padding are not integers.

        Examples:
            >>> conv = Convolution(padding=1, stride=1, dilation=1)
            >>> conv.run(matrix, kernel)

        Returns:
            Convolution:    Convolution object.
        """
        # Extracting padding and transforming it to tuple if it is an integer
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

        # Extracting stride and transforming it to tuple if it is an integer
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        # Extracting dilation and transforming it to tuple if it is an integer
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.dilation = dilation

        # Check if parameters are correct
        self._check_params()

    @profile
    def _add_padding(self,
                     matrix: np.ndarray) -> np.ndarray:
        """Adds padding to the matrix/image.

        Args:
            matrix (np.ndarray): See `self.run` method.

        Returns:
            np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
        """
        # Extract padding values
        row, col = self.padding

        # Extract rows and columns for image
        matrix_n, matrix_m = matrix.shape

        # Add padding to the matrix
        padded_matrix = np.zeros((matrix_n + row * 2, matrix_m + col * 2))
        padded_matrix[row : matrix_n + row, col : matrix_m + col ] = matrix

        return padded_matrix

    @profile
    def _check_params(self) -> np.ndarray:
        """Internal function to check if the input parameters are correct.

        Raises:
            ValueError: If parameters are not integers or are less than default values.
        """
        # Check if parameters are correct
        
        params_are_correct = all(
            isinstance(param, int) and param >= 1
            for param in [*self.stride, *self.dilation]
        ) and all(
            isinstance(param, int) and param >= 0
            for param in self.padding
        )

        if not params_are_correct:
            raise ValueError('Parameters should be integers equal or greater than default values.')

    @profile
    def _check_matrix_and_kernel(self,
                                matrix: Union[List[List[float]], np.ndarray],
                                kernel: Union[List[List[float]], np.ndarray]
                                ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], int, int]:
        """Internal function to check if the matrix and kernel are correct.

        Args:
            matrix (Union[List[List[float]], np.ndarray]): See `self.run` method.
            kernel (Union[List[List[float]], np.ndarray]):  See `self.run` method.

        Returns:
            Tuple[np.ndarray, np.ndarray, Tuple[int, int], int, int]: Padded matrix, kernel,
                kernel shape, height of the output matrix, width of the output matrix.
        """
        # Check if matrix and kernel are numpy arrays
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)

        # Extract rows and columns for image and kernel
        matrix_n, matrix_m = matrix.shape

        # Add padding to the matrix
        matrix = matrix if list(self.padding) == [0, 0] else self._add_padding(matrix=matrix)

        # Extracting new rows and columns for image with padding added
        n_p, m_p = matrix.shape

        # Validate kernel
        kernel_shape = self._kernel_checker(kernel=kernel, n_p=n_p, m_p=m_p)

        h_out, w_out = self._out_dimensions(matrix_n=matrix_n, matrix_m=matrix_m,
                                            kernel_shape=kernel_shape)

        return matrix, kernel, kernel_shape, h_out, w_out

    @profile
    def _kernel_checker(self,
                        kernel: Union[List[List[float]], np.ndarray],
                        n_p: int,
                        m_p: int):
        """Internal function to check if the kernel is correct.

        Args:
            kernel (Union[List[List[float]], np.ndarray]):  see `self.run` method.
            n_p (int):  Number of rows in the matrix with padding.
            m_p (int):  Number of columns in the matrix with padding.
        """
        # Check if kernel is numpy array
        if not isinstance(kernel, np.ndarray):
            kernel = np.array(kernel)

        # Extract rows and columns for kernel
        kernel_shape = kernel.shape

        # Validating kernel shape
        kernel_is_correct = kernel_shape[0] % 2 == 1 and kernel_shape[1] % 2 == 1
        if not kernel_is_correct:
            raise ValueError(f'Kernel shape should be odd. Kernel shape is: {kernel_shape}')

        # Validating kernel size
        matrix_to_kernel_is_correct = n_p >= kernel_shape[0] and m_p >= kernel_shape[1]
        if not matrix_to_kernel_is_correct:
            raise ValueError('Kernel can\'t be bigger than matrix in terms of shape.')

        return kernel_shape

    @profile
    def _out_dimensions(self,
                        matrix_n: int,
                        matrix_m: int,
                        kernel_shape: Tuple[int, int]):
        """Calculates the output dimensions of the convolution operation.

        Args:
            n (int):    Number of rows in the matrix.
            m (int):    Number of columns in the matrix.
            k (Tuple[int, int]):    Tuple with number of rows and columns in the kernel.
        """
        # Calculate output dimensions
        h_out = np.floor((matrix_n + 2 * self.padding[0] - kernel_shape[0] -
                          (kernel_shape[0] - 1) * (self.dilation[0] - 1)) /
                         self.stride[0]).astype(int) + 1

        w_out = np.floor((matrix_m + 2 * self.padding[1] - kernel_shape[1] -
                          (kernel_shape[1] - 1) * (self.dilation[1] - 1)) /
                         self.stride[1]).astype(int) + 1

        # Validate that output dimensions are greater than 0
        out_dimensions_are_correct = h_out > 0 and w_out > 0
        if not out_dimensions_are_correct:
            raise ValueError(('Can\'t apply input parameters, one of resulting output dimension '
                              'is non-positive.'))

        return h_out, w_out

    @profile
    def run(self,
            matrix: Union[List[List[float]], np.ndarray],
            kernel: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and
        padding along axes.

        Args:
            matrix (Union[List[List[float]], np.ndarray]):  2D matrix to be convolved.
            kernel (Union[List[List[float]], np.ndarray]):  2D odd-shaped matrix
                (e.g. 3x3, 5x5, 13x9, etc.).
        Returns:
            np.ndarray: 2D Feature map, i.e. matrix after convolution.
        """

        # Validate parameters
        matrix, kernel, k, h_out, w_out = self._check_matrix_and_kernel(matrix, kernel)

        # Initialize output matrix
        matrix_out = np.zeros((h_out, w_out))

        # Extract rows and columns for image and kernel
        kernel_center = k[0] // 2, k[1] // 2

        # Get x0 and y0
        center_x_0 = kernel_center[0] * self.dilation[0]
        center_y_0 = kernel_center[1] * self.dilation[1]

        # Loop over height of the matrix

        for i in range(h_out):
            # Calculate the x-axis center for the kernel
            center_x = center_x_0 + i * self.stride[0]

            # Calculate indices for the kernel
            indices_x = [center_x + l * self.dilation[0] for l in range(-kernel_center[0],
                                                                        kernel_center[0] + 1)]

            # Loop over width of the matrix
            for j in range(w_out):
                # Calculate the y-axis center for the kernel
                center_y = center_y_0 + j * self.stride[1]

                # Calculate indices for the kernel
                indices_y = [center_y + l * self.dilation[1] for l in range(-kernel_center[1],
                                                                            kernel_center[1] + 1)]

                # Extract submatrix from the matrix
                submatrix = matrix[indices_x, :][:, indices_y]

                # Perform convolution operation and store the result in the output matrix
                matrix_out[i][j] = np.sum(np.multiply(submatrix, kernel))

        # Return the output matrix
        return matrix_out
<write execution script for convolution>
