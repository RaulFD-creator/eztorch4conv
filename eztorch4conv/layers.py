from abc import abstractclassmethod
import torch.nn as nn
from torch.nn import Flatten
from torch.nn import Sigmoid

class layer():
    def __init__(self, input_shape=None, **kwargs):

        """
        Input shape should be a vector (n_channels, x, y, z)
        """
        self.input_shape = input_shape
        try:
            self.out_channels = kwargs['neurons']
        except KeyError:
            raise Exception('Need to specify how many neurons in each layer')
        try:
            self.activation_function = kwargs['activation_function']
        except KeyError:
            self.activation_function = nn.ELU()
        try:
            self.dropout_proportion = kwargs['dropout']
        except KeyError:
            self.dropout_proportion = None
        
    def create_dropout(self):
        if self.dropout_proportion is not None:
            self.dropout = nn.Dropout(self.dropout_proportion)
        else:
            self.dropout = nn.Dropout(0)
            
    @abstractclassmethod
    def create_pooling(self):
        "For custom layers, this method has to be explictly programmed"
        
    @abstractclassmethod
    def create_main_layer(self):
        "For custom layers, this method has to be explictly programmed"
    
    @abstractclassmethod
    def build_layer(self):
        self.create_main_layer()
        self.create_dropout()
        return nn.Sequential(self.main_layer, self.dropout, self.activation_function)
        
class conv3d(layer):
    def __init__(self, input_shape=None, **kwargs):
        super().__init__(input_shape, kwargs)
        try:
            self.batch_norm = kwargs['batch_norm']
        except KeyError:
            self.batch_norm = None
        try:
            self.conv_kernel = kwargs['conv_kernel']
        except KeyError:
            raise Exception('Need to provide an specific kernel shape for conv layer')
        try:
            self.conv_padding = kwargs['conv_padding']
        except KeyError:
            self.conv_padding = 0
        try:
            self.pooling_type = kwargs['pooling_type']
        except KeyError:
            self.pooling_type = None
        try:
            self.pooling_kernel = kwargs['pooling_kernel']
        except KeyError:
            if self.pooling_type is not None:
                self.pooling_kernel = 2
            else:
                self.pooling_kernel = None
                     
    def create_pooling(self):
        if str(self.pooling_type).lower() != "none" and self.pooling_type is not None:
            if self.pooling_type.lower() == "max":
                self.pooling = nn.MaxPool3d(self.pooling_kernel)
            elif self.pooling_type.lower() == "avg":
                self.pooling = nn.AvgPool3d(self.pooling_kernel)
        else:
            self.pooling = None
     
    def build_layer(self):
        self.create_main_layer()
        self.create_dropout()
        self.create_pooling()
        if self.pooling is not None and self.batch_norm is None:
            return nn.Sequential(self.main_layer, self.dropout, self.activation_function, self.pooling)
        elif self.pooling is not None and self.batch_norm is not None:
            return nn.Sequential(self.main_layer, self.batch_norm, self.dropout, self.activation_function, self.pooling)
        elif self.pooling is None and self.batch_norm is not None:
            return nn.Sequential(self.main_layer, self.batch_norm, self.dropout, self.activation_function)
        else:
            return nn.Sequential(self.main_layer, self.dropout, self.activation_function)
    
    def create_main_layer(self):
        self.in_channels = self.input_shape[0]
        self.main_layer = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel_size=self.conv_kernel, padding=self.conv_padding)
    
    def calculate_output_shape(self):
        n_channels = self.out_channels
        x = self.input_shape[1] - (2 ) * (self.conv_kernel - (self.conv_padding + 2))
        y = self.input_shape[2] - (2 ) * (self.conv_kernel - (self.conv_padding + 2))
        z = self.input_shape[3] - (2 ) * (self.conv_kernel - (self.conv_padding + 2))
        return (n_channels, x, y, z)

class dense(layer):
    def __init__(self, input_shape, **kwargs):
        super().__init__(input_shape, kwargs)
        self.out_channels = self.neurons

    def create_main_layer(self):
        self.in_channels = self.input_shape
        self.main_layer = nn.Linear(in_features=self.in_channels, out_features=self.out_channels)

    def calculate_output_shape(self):
        return self.out_channels

class flatten():    
    def build_layer(self):
        return nn.Flatten(1,-1)

    def calculate_output_shape(self):
        return  self.input_shape[0]*self.input_shape[1]*self.input_shape[2]*self.input_shape[3]
    
class sigmoid(Sigmoid):
    def build_layer(self):
        return self

