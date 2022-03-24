import torch.nn as nn
from torch.nn import Flatten
from torch.nn import Sigmoid

class layer():
    def __init__(self, model, neurons, conv_kernel=3, conv_padding=1, 
                activation_function = nn.LeakyRelU(), pooling_type='max', pooling_kernel=2, 
                dropout=None):
        
        self.out_channels = neurons
        self.conv_kernel = conv_kernel
        self.conv_padding = conv_padding
        self.activation_function = activation_function
        self.pooling_type = pooling_type
        self.pooling_kernel = pooling_kernel
        self.model = model
        self.dropout = dropout
        self.pooling_type = pooling_type
        self.dropout = dropout
        
        self.layer = nn.ModuleList()

    def create_dropout(self):
        if self.dropout:
            self.dropout = nn.Dropout(self.dropout)
            self.layer.append(self.dropout)

    def create_pooling(self):
        if str(self.pooling_type).lower() != "none" and self.pooling.type is not None:
            if self.pooling_type.lower() == "max":
                self.layer.append(nn.MaxPool3d(self.pooling_kernel))
            elif self.pooling_type.lower() == "avg":
                self.layer.append(nn.AvgPool3d(self.pooling_kernel))
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
            self.create_dropout()
            self.create_activation_function()
            self.create_pooling()
        else:
            self.create_main_layer(self.model.channels)
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
    
    def create_main_layer(self, in_channels):
        self.layer.append(nn.Conv3d(in_channels=in_channels, out_channels=self.out_channels,
                                    kernel_size=self.conv_kernel, padding=self.conv_padding))
    

class dense(layer):
    def __init__(self, out_features, dropout=None):

        self.out_features = out_features
        self.dropout = dropout
        
    def create_main_layer(self, in_channels):
        self.layer.append(nn.Linear(in_channels=in_channels, out_channels=self.out_channels))

class flatten(Flatten):    
    def build_layer(self, prev_channels):
        return self
    
class sigmoid(Sigmoid):
    def build_layer(self, prev_channels):
        return self
