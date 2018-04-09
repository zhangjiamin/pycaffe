import numpy
from layer import Layer
from blob import Blob

class BaseConvolutionLayer(Layer):

    def __init__(self):
        self.kernel_shape_ = None
        self.stride_       = None
        self.pad_          = None
        self.dilation_     = None
        self.conv_input_shape_ = None
        self.col_buffer_shape_ = None
        self.output_shape_     = None
        self.bottom_shape_     = None

        self.num_spatial_axes_ = None
        self.bottom_dim_       = None
        self.top_dim_          = None
        self.channel_axis_     = None
        self.num_              = None
        self.channels_         = None
        self.group_            = None
        self.out_spatial_dim_  = None
        self.weight_offset_    = None
        self.num_output_       = None
        self.bias_term_        = None
        self.is_1x1_           = None
        self.force_nd_im2col_  = None

    def LayerSetup(self, bottom, top):
        pass

    def Reshape(self, bottom, top):
        pass

    def MinBottomBlobs(self):
        return 1

    def MinTopBlobs(self):
        return 1

    def EqualNumBottomTopBlobs(self):
        return True

    def forward_cpu_gemm(self, input, weights, output, skip_im2col=False):
        pass

    def forward_cpu_bias(self, output, bias):
        pass

    def backward_cpu_gemm(self, input, weights, output):
        pass

    def weight_cpu_gemm(self, input, output, weights):
        pass

    def backward_cpu_bias(self, bias, input):
        pass

    def input_shape(self, i):
        pass

    def reverse_dimensions(self):
        pass

    def compute_output_shape(self):
        pass

    

