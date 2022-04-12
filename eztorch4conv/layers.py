from abc import abstractmethod
import torch.nn as nn

class layer():
    def __init__(self, input_shape=None, **kwargs):
        """
        Input shape should be a vector (n_channels, x, y, z)
        """
        self.input_shape = input_shape
        self.modules = []

        try:
            self.out_channels = kwargs['neurons']
        except KeyError:
            raise Exception('Need to specify how many neurons comprise each layer')
        try:
            self.activation_function = kwargs['activation_function']
        except KeyError:
            self.activation_function = nn.ELU()
        try:
            self.dropout_proportion = kwargs['dropout']
        except KeyError:
            self.dropout_proportion = None
        
    def _create_dropout(self):
        if self.dropout_proportion is not None:
            self.modules.append(nn.Dropout(self.dropout_proportion))
        else:
            self.modules.append(nn.Dropout(0))
    
    def _create_activation_function(self):
        self.modules.append(self.activation_function)

    @abstractmethod
    def _create_pooling(self):
        "For custom layers, this method has to be explictly programmed"
    
    @abstractmethod
    def _create_main_layer(self):
        "For custom layers, this method has to be explictly programmed"

    @abstractmethod
    def _create_batch_norm(self):
        "For custom layers, this method has to be explictly programmed"

    def build_layer(self):
        self._create_batch_norm()
        self._create_main_layer()
        self._create_activation_function()
        self._create_dropout()
        self._create_pooling()
        return nn.Sequential(*self.modules)
        
class conv3d(layer):
    def __init__(self, input_shape=None, **kwargs):   

        arguments = kwargs.keys()     
        if 'activation_function' not in arguments:
            kwargs['activation_function'] = nn.ELU()

        if 'dropout' not in arguments:
            kwargs['dropout'] = 0

        if 'batch_norm' not in arguments:
            self.batch_norm = False
        else:
            self.batch_norm = kwargs['batch_norm']

        if 'conv_kernel' in arguments:
            self.conv_kernel = kwargs['conv_kernel']
        else:
            raise Exception('Need to provide an specific kernel shape for conv layer')

        if 'padding' in arguments:
            self.padding = kwargs['padding']
        else:
            self.padding = 'same'

        if 'pooling_type' in arguments:
            self.pooling_type = kwargs['pooling_type']
        else:
            self.pooling_type = None

        if 'pooling_kernel' in arguments:
            self.pooling_kernel = kwargs['pooling_kernel']
        else:
            self.pooling_kernel = 2
                 
        super().__init__(input_shape, 
                        neurons=kwargs['neurons'], 
                        activation_function=kwargs['activation_function'],
                        dropout = kwargs['dropout']
                        )

    def _create_pooling(self):
        if str(self.pooling_type).lower() != "none":
            if self.pooling_type.lower() == "max":
                self.modules.append(nn.MaxPool3d(kernel_size=self.pooling_kernel,stride=self.pooling_kernel))
            elif self.pooling_type.lower() == "avg":
                self.modules.append(nn.AvgPool3d(kernel_size=self.pooling_kernel,stride=self.pooling_kernel))
    
    def _create_batch_norm(self):
        self.in_channels = self.input_shape[0]

        if self.batch_norm:
            self.modules.append(nn.BatchNorm3d(self.in_channels))
        
    def _create_main_layer(self):
        self.in_channels = self.input_shape[0]
        if self.padding == 'valid':
            self.conv_padding = 0
        elif self.padding == 'same':
            self.conv_padding = self.conv_kernel // 2 
        elif self.padding is int:
            self.conv_padding = self.padding
        
        self.modules.append(nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel_size=self.conv_kernel, padding=self.conv_padding))
    
    def calculate_output_shape(self):
        n_channels = self.out_channels


        x = self.input_shape[1] - self.conv_kernel + 2 * self.conv_padding + 1 
        y = self.input_shape[2] - self.conv_kernel + 2 * self.conv_padding + 1
        z = self.input_shape[3] - self.conv_kernel + 2 * self.conv_padding + 1 

        if str(self.pooling_type).lower() != "none":
            x = x // self.pooling_kernel 
            y = y // self.pooling_kernel
            z = z // self.pooling_kernel 

        return (n_channels, x, y, z)

    
class conv2d(conv3d):

    def _create_batch_norm(self):
        self.in_channels = self.input_shape[0]

        if self.batch_norm:
            self.modules.append(nn.BatchNorm2d(self.in_channels))

    def _create_main_layer(self):
        
        self.in_channels = self.input_shape[0]

        if self.padding == 'valid':
            self.conv_padding = 0
        elif self.padding == 'same':
            self.conv_padding = self.conv_kernel // 2
        elif self.padding is int:
            self.conv_padding = self.padding
        
        self.modules.append(nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel_size=self.conv_kernel, padding=self.conv_padding))

    def calculate_output_shape(self):
        n_channels = self.out_channels
        x = self.input_shape[1] - self.conv_kernel + 2 * self.conv_padding + 1
        y = self.input_shape[2] - self.conv_kernel + 2 * self.conv_padding + 1
        return (n_channels, x, y)
    
class dense(layer):
    def __init__(self, input_shape=None, **kwargs):
               
        try:
            a = kwargs['activation_function']
        except KeyError:
            kwargs['activation_function'] = nn.ELU()
        try:
            a = kwargs['dropout']
        except KeyError:
            kwargs['dropout'] = 0
         
        super().__init__(input_shape, neurons=kwargs['neurons'], activation_function=kwargs['activation_function'],
                        dropout = kwargs['dropout'])
        
        self.out_channels = kwargs['neurons']

    def _create_main_layer(self):
        self.in_channels = self.input_shape
        self.modules.append(nn.Linear(in_features=self.in_channels, out_features=self.out_channels))

    def calculate_output_shape(self):
        return self.out_channels

class flatten():
    def __init__(self):
        self.input_shape = []

    def build_layer(self):
        return nn.Flatten(1,-1)

    def calculate_output_shape(self):
        return  self.input_shape[0] * self.input_shape[1] * self.input_shape[2] * self.input_shape[3]
