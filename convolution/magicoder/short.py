import numpy as np
from scipy import signal

class ConvolutionLayer:
    def __init__(self, filter_size, stride=1, padding='valid', dilation_rate=1):
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.dilation_rate = dilation_rate

    def create_filters(self, channel_num):
        return np.random.randn(self.filter_size, self.filter_size, channel_num) / 9

    def zero_padding(self, image):
        if self.padding == 'valid':
            return image
        else:
            pad_width = ((0,0), (self.filter_size-1,self.filter_size-1),
                         (self.filter_size-1,self.filter_size-1))
            return np.pad(image, pad_width=pad_width)

    def forward(self, image):
        # Adding padding to the input image if required
        padded_image = self.zero_padding(image)

        output_shape = ((padded_image.shape[0] - self.filter_size + 1) // self.stride, (padded_image.shape[1] -
                                                                                        self.filter_size + 1) // self.stride)

        # Create filters if not already created
        if not hasattr(self, 'filters'):
            channel_num = padded_image.shape[-1] if len(padded_image.shape) > 2 else 1
            self.filters = self.create_filters(channel_num)

        output = np.zeros((output_shape[0], output_shape[1]))

        for i in range(0, padded_image.shape[0] - self.filter_size + 1, self.stride):
            for j in range(0, padded_image.shape[1] - self.filter_size + 1, self.stride):
                output[i // self.stride][j // self.stride] = np.sum(padded_image[i: i + self.filter_size, j: j
                                                                                 + self.filter_size] * self.filters)

        return output
