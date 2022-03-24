import torch.nn as nn
from torch.nn import Flatten
from torch.nn import Sigmoid

class layer():
    def __init__(self, model, neurons, conv_kernel, conv_padding=1, 
                activation_function = nn.LeakyRelu(), pooling_type='max', pooling_kernel=2, 
                dropout=0.25):
        
        self.out_channels = neurons
        self.conv_kernel = conv_kernel
        self.conv_padding = conv_padding
        self.activation_function = activation_function
        self.pooling_type = pooling_type
        self.pooling_kernel = pooling_kernel
        self.model = model
        self.dropout = dropout

        self.layer = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        if str(pooling_type).lower() != "none":
            if pooling_type.lower() == "max":
                self.pooling = nn.MaxPool3d(pooling_kernel)
            elif pooling_type.lower() == "avg":
                self.pooling = nn.AvgPool3d(pooling_kernel)
        else:
            self.pooling = None
    

    def create_main_layer(self, in_channels):
        self.layer.append(nn.Conv3d(in_channels=in_channels, out_channels=self.out_channels,
                                    kernel_size=self.conv_kernel, padding=self.conv_padding))
    
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

    def build_layer(self, prev_layer):
        if prev_layer:
            self.create_main_layer(prev_layer.out_channels)
        else:
            self.create_main_layer(self.model.channels)



class conv3d():
    def __init__(self, in_channels, out_channels, conv_kernel, conv_padding=1, 
                activation_function = nn.LeakyRelu(), pooling_type='max', pooling_kernel=2, 
                dropout=0.25):
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel = conv_kernel
        self.conv_padding = conv_padding
        self.activation_function = activation_function
        self.pooling_type = pooling_type
        self.pooling_kernel = pooling_kernel
        self.dropout = dropout

        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=conv_kernel, padding=conv_padding)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        if str(pooling_type).lower() != "none":
            if pooling_type.lower() == "max":
                self.pooling = nn.MaxPool3d(pooling_kernel)
            elif pooling_type.lower() == "avg":
                self.pooling = nn.AvgPool3d(pooling_kernel)
        else:
            self.pooling = None

    def build_layer(self):
        if self.pooling is not None:
            return nn.Sequential(self.conv, self.leaky_relu, self.pooling, self.dropout)
        else:
            return nn.Sequential(self.conv, self.leaky_relu, self.dropout)

class dense():
    def __init__(self, in_features, out_features, dropout=0.4):

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def build_layer(self):
            return nn.Sequential(self.linear, self.leaky_relu, self.dropout)

class flatten(Flatten):
    
    def build_layer(self):
        return self
    
class sigmoid(Sigmoid):
    def build_layer(self):
        return self
