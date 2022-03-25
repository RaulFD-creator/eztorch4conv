import torch.nn as nn
from torch.nn import Flatten
from torch.nn import Sigmoid

class layer():
    def __init__(self, neurons, conv_kernel=3, conv_padding=1, 
                activation_function = nn.LeakyReLU(), pooling_type='max', pooling_kernel=2, 
                dropout=None, input_shape=None):

        """
        Input shape should be a vector (n_channels, x, y, z)
        """
        
        self.input_shape = input_shape
        self.out_channels = neurons
        self.conv_kernel = conv_kernel
        self.conv_padding = conv_padding
        self.activation_function = activation_function
        self.pooling_type = pooling_type
        self.pooling_kernel = pooling_kernel
        self.dropout = dropout
        self.pooling_type = pooling_type
        self.dropout = dropout
        
        self.layer = nn.ModuleList()

    def create_dropout(self):
        if self.dropout:
            self.dropout = nn.Dropout(self.dropout)
            self.layer.append(self.dropout)

    def create_pooling(self):
        print("For custom layers, this method has to be explictly programmed")
    
    def create_main_layer(self):
        print("For custom layers, this method has to be explictly programmed")

    def create_activation_function(self):
        if isinstance(self.activation_function, str):
            self.activation_functions = {
                'leakyrelu': nn.LeakyReLU(),
                'relu': nn.ReLU(),
                'elu': nn.ELU(),
                'hardshrink': nn.Hardshrink(),
                'hardsigmoid': nn.Hardsigmoid(),
                'hardtanh': nn.Hardtanh(),
                'hardswish': nn.Hardswish(),
                'logsigmoid': nn.LogSigmoid(),
                'multiheadattention': nn.MultiheadAttention(),
                'prelu': nn.PreLU(),
                'relu6': nn.ReLU6(),
                'rrelu': nn.RReLU(),
                'selu': nn.SELU(),
                'celu': nn.CELU(),
                'gelu': nn.GELU(),
                'sigmoid': nn.Sigmoid(),
                'silu': nn.SiLU(),
                'mish': nn.Mish(),
                'softplus': nn.Softplus(),
                'softshrink': nn.Softshrink(),
                'softsign': nn.Softsign(),
                'tanh': nn.Tanh(),
                'tabhshrink': nn.Tanhshrink(),
                'threshold': nn.Threshold(),
                'glu': nn.GLU(), 
                'softmin': nn.Softmin(),
                'softmax': nn.Softmax(), 
                'softmax': nn.Softmax2d(),
                'logsoftmax': nn.LogSoftmax(),
                'adaptivelogsoftmaxwithloss': nn.AdaptiveLogSoftmaxWithLoss()
            }
            self.activation_function = self.activation_functions[self.activation_function.lower()]
            del(self.activation_functions)
        self.layer.append(self.activation_function)

    def build_layer(self):
        self.create_main_layer()
        self.create_dropout()
        self.create_activation_function()
        self.create_pooling()
        return self.layer

class conv3d(layer):
    def create_pooling(self):
        if str(self.pooling_type).lower() != "none" and self.pooling.type is not None:
            if self.pooling_type.lower() == "max":
                self.layer.append(nn.MaxPool3d(self.pooling_kernel))
            elif self.pooling_type.lower() == "avg":
                self.layer.append(nn.AvgPool3d(self.pooling_kernel))
        else:
            self.pooling = None
    
    def create_main_layer(self):
        self.layer.append(nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel_size=self.conv_kernel, padding=self.conv_padding))
    
    def calculate_output_shape(self):
        n_channels = self.input_shape[0]
        x = self.input_shape[1] - 2 * 3 ** (self.conv_kernel - (self.conv_padding + 2))
        y = self.input_shape[2] - 2 * 3 ** (self.conv_kernel - (self.conv_padding + 2))
        z = self.input_shape[3] - 2 * 3 ** (self.conv_kernel - (self.conv_padding + 2))
        return (n_channels, x, y, z)

class dense(layer):
    def __init__(self,  out_features, dropout=None, in_channels=None):

        self.in_channels = in_channels
        self.out_features = out_features
        self.dropout = dropout
        
    def create_main_layer(self):
        self.layer.append(nn.Linear(in_channels=self.in_channels, out_channels=self.out_channels))

class flatten(Flatten):    
    def build_layer(self):
        return self

    def calculate_output_shape(self):
        return self.input_shape[0] * self.input_shape[1] * self.input_shape[2] * self.input_shape[3]
    
class sigmoid(Sigmoid):
    def build_layer(self):
        return self
