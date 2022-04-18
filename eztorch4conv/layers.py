# Copyright by Raúl Fernández Díaz

"""
Layers module within the eztorch4conv library. It contains the basic class blueprint
for constructing new layers for DCNN and MC-DCNN models, and 3 of the most common
implementations: conv3d that includes all the elements that may be needed for a 3D convolutional
layer, dense that includes all the elements that may be needed for a dense layer, and flatten
that incorporates an eztorch4conv-compatible sintaxis for the Pytorch flatten layer.
"""

from abc import abstractmethod
import torch.nn as nn

class layer():
    """
    Basic class blueprint for easy construction of eztorch4conv layers.

    Methods
    -------
    build_layer() : Creates a torch.nn.Sequential object that contains
                    all the different objects that conform the layer.

    calculate_output_shape() : Returns an int or tuple with the dimensions of
                                the output of the layer to facilitate
                                the automatic creation of the next layer in 
                                the model.

    Attributes
    ----------
    activation_function : torch activation function object
                Torch activation function object
                By default : torch.nn.ELU()
    dropout : float
        Proportion of neurons to be randomly dropout
    input_shape : tuple
            Tuple with the dimensions of the input data with shape
            (n_channels, dimensions) in the case of 3D convolution it
            will be (n_channels, x, y, z)
    modules : list 
        Inner list that accumulates all the objects that will conform
        the layer
    neurons : int
        Number of neurons that will compose the layer
    """
    def __init__(self, input_shape=None, **kwargs):
        """
        Instanciates the class object.

        Parameters
        ----------
        activation_function : torch activation function object
            Torch activation function object
            By default : torch.nn.ELU()
        dropout : float
            Proportion of neurons to be randomly dropout.
            By default : 0
        input_shape : tuple
                Tuple with the dimensions of the input data with shape
                (n_channels, x, y, z)
        neurons : int
            Number of neurons that will compose the layer

        Returns
        -------
        layer : layer class instance
        """
        self.input_shape = input_shape
        self.modules = []

        # Check all input arguments and set default values if needed. 
        # Handle any errors that may arise.
        try: self.neurons = kwargs['neurons']
        except KeyError: raise KeyError('Please specify how many neurons comprise each layer')
        
        try: self.activation_function = kwargs['activation_function']
        except KeyError: self.activation_function = nn.ELU()
        
        try: self.dropout = kwargs['dropout']
        except KeyError: self.dropout = None
        
    def _create_dropout(self):
        """
        Helper function to build_layer(). It creates the dropout object
        and appends it to the modules list.
        """
        # Create dropout layer only if self.dropout is not None
        if str(self.dropout).lower() != "none": self.modules.append(nn.Dropout(self.dropout))
    
    def _create_activation_function(self):
        """
        Helper function to build_layer(). It appends the activation
        function object to the modules list.
        """
        self.modules.append(self.activation_function)

    @abstractmethod
    def _create_pooling(self):
        """
        Helper function to build_layer(). It creates a pooling object and appends it to
        the modules list. It is only necessary in convolutional layers and has to be explictly 
        programmed for custom layers.
        """
    
    @abstractmethod
    def _create_main_layer(self): 
        """       
        Helper function to build_layer(). It creates the main layer object and appends it
        to the modules list. This method has to be explictly programmed for custom layers.
        """

    @abstractmethod
    def _create_batch_norm(self):
        """
        Helper function to build_layer(). It creates a batch normalization object and appends it to
        the modules list. It is only necessary in convolutional layers and has to be explictly 
        programmed for custom layers.
        """

    @abstractmethod
    def calculate_output_shape(self):
        """
        Computes the shape of the output of the layer to facilitate the automatic
        creation of subsequent layers. Has to be explictly programmed for custom
        layers.

        Returns
        -------
        output_shape : int or tuple
                    Dimensions of the output of the layer
        """

    def build_layer(self):
        """
        Creates a torch.nn.Sequential object from the modules list that includes all the predefined
        objects that the layer class requires.

        Returns
        -------
        layer : torch.nn.Sequential()
            torch.nn.Sequential object that contains all the predefined objects required by the layer class
        """
        # Order of the layers is important, only one that could be moved is batch_norm, though
        # literature seems to agree it is best if it is the very first layer of the model.
        self._create_batch_norm()
        self._create_main_layer()
        self._create_activation_function()
        self._create_dropout()
        self._create_pooling()
        return nn.Sequential(*self.modules)
        
class conv3d(layer):
    """
    3D Convolutional layer implementation in eztorch4conv environment. 
    It includes easy creation of dropout, batch normalization, and 
    various kinds of pooling. It inherits the methods and attributes from the layer class.

    Attributes
    ----------
    batch_norm : bool
            Flag that indicates whether there should be batch normalization.
            By default : False
    conv_kernel : int or tuple
            Indicates the shape or dimensions of the convolutional kernel applied.
            If int the shape of the kernel will be cubic.
    padding : str or int
            Indicates whether padding should be added so as to not reduce the 
            size of the output during the convolution ('same') or if should 
            not be introduced ('valid'). If int is provided, the padding will
            incorporate as many zeros as the padding value indicates.
            By default : 'same'
    pooling_kernel : int or tuple
                Indicates the shape or dimensions of the pooling kernel applied.
                If int the shape of the kernel will be cubic.
                By default : 2    
    pooling_stride : int
                Indicates how the pooling kernel will move through the image.
                By default : pooling_kernel
    pooling_type : str
            Indicates the type of pooling that should be introduced, if any.
            Options are: max and average.
            By defauld : None
    """
    def __init__(self, input_shape=None, **kwargs):  
        """
        Instanciates the class object. Parameters are to be introduced
        explictily as if they were keys in a dictionary. 
        
        Parameters
        ----------
        activation_function : torch activation function object
                        Torch activation function object
                        By default : torch.nn.ELU()
        batch_norm : bool
                Flag that indicates whether there should be batch normalization.
                By default : False
        conv_kernel : int or tuple
                Indicates the shape or dimensions of the convolutional kernel applied.
                If int the shape of the kernel will be cubic.
        dropout : float
            Proportion of neurons to be randomly dropout.
            By default : 0
        input_shape : tuple
                Tuple with the dimensions of the input data with shape
                (n_channels, x, y, z). Has only to be introduced in the
                first layer of the model or in the case of MC-DCNN
                in the first layer of a channel
        neurons : int
            Number of neurons that will compose the layer
        padding : str
            Indicates whether padding should be added so as to not reduce the 
            size of the output during the convolution ('same') or if should 
            not be introduced ('valid').
            By default : 'same'
        pooling_kernel : int or tuple
                    Indicates the shape or dimensions of the pooling kernel applied.
                    If int the shape of the kernel will be cubic.
                    By default : 2    
        pooling_stride : int
                    Indicates how the pooling kernel will move through the image.
                    By default : pooling_kernel
        pooling_type : str
                Indicates the type of pooling that should be introduced, if any.
                Options are: max and average.
                By defauld : None

        Returns
        -------
        conv3d : conv3d class instance

        Example
        -------
        >>> ez.layers.conv3d(input_shape=(6,16,16,16), neurons=32, 
                conv_kernel=5, padding='same', pooling_type='max', 
                dropout=0, batch_norm=True)
        """ 
        # Check all input argumetns and set the default values if needed. 
        # Handle any errors that may arise.
        arguments = kwargs.keys()     

        if 'neurons' not in arguments: raise KeyError('Please specify how many neurons comprise each layer')

        if 'activation_function' not in arguments: kwargs['activation_function'] = nn.ELU()

        if 'dropout' not in arguments: kwargs['dropout'] = 0

        if 'batch_norm' not in arguments: self.batch_norm = False
        else:
            if kwargs['batch_norm'] == True or kwargs['batch_norm'] == False: self.batch_norm = kwargs['batch_norm'] 
            else: raise ValueError(f"Value {kwargs['batch_norm']} is not an available option of batch_norm.\nbatch_norm has to be either True or False.")

        if 'conv_kernel' in arguments: self.conv_kernel = kwargs['conv_kernel']
        else: raise KeyError('Please provide an specific kernel shape for conv layer')

        if 'padding' in arguments: self.padding = kwargs['padding']
        else: self.padding = 'same'

        if 'pooling_type' in arguments: self.pooling_type = kwargs['pooling_type']
        else: self.pooling_type = None

        if 'pooling_kernel' in arguments: self.pooling_kernel = kwargs['pooling_kernel']
        else: self.pooling_kernel = 2

        if 'pooling_stride' in arguments: self.pooling_stride = kwargs['pooling_stride']
        else: self.pooling_stride = self.pooling_kernel
        
        # Pass rest of the arguments to the __init__ function of parent class
        super().__init__(input_shape, 
                        neurons=kwargs['neurons'], 
                        activation_function=kwargs['activation_function'],
                        dropout = kwargs['dropout']
                        )

    def _create_pooling(self):
        """
        Helper function to build_layer(). It creates the appropriate pooling layer
        and appends it to the modules list.
        """
        # Create appropriate pooling layer only if self.pooling_type is not None
        if str(self.pooling_type).lower() != "none":
            if self.pooling_type.lower() == "max":
                self.modules.append(nn.MaxPool3d(kernel_size=self.pooling_kernel,stride=self.pooling_stride))
            elif self.pooling_type.lower() == "avg":
                self.modules.append(nn.AvgPool3d(kernel_size=self.pooling_kernel,stride=self.pooling_stride))
    
    def _create_batch_norm(self):
        """
        Helper function to build_layer(). It creates the appropriate batch normalization layer
        and appends it to the modules list.
        """
        # Create a batch normalization layer that considers as many inputs as outputs had the previous
        # layer
        self.in_channels = self.input_shape[0]

        if self.batch_norm: self.modules.append(nn.BatchNorm3d(self.in_channels))
        
    def _create_main_layer(self):
        """
        Helper function to build_layer(). It creates the main 3D convolutional layer, with
        the appropriate padding and appends it to the modules list.
        """
        # Create the 3D convolutional layer with as many inputs as outputs the previous layers
        # had and with the appropriate padding style.
        self.in_channels = self.input_shape[0]

        if self.padding == 'valid': self.conv_padding = 0
        elif self.padding == 'same': self.conv_padding = self.conv_kernel // 2 
        elif isinstance(self.padding, int): self.conv_padding = self.padding
        
        self.modules.append(nn.Conv3d(in_channels=self.in_channels, out_channels=self.neurons,
                                    kernel_size=self.conv_kernel, padding=self.conv_padding))
    
    def calculate_output_shape(self):
        """
        Calculates the output shape of the layer to facilitate the creation of 
        subsequent layers of the model.

        Returns
        -------
        output_shape : tuple
                    Tuple with the dimensions of the output of the layer,
                    (n_channels, x, y, z)
        """
        n_channels = self.neurons

        # Calculate the effect of padding and convolution on output size
        x = self.input_shape[1] - self.conv_kernel + 2 * self.conv_padding + 1 
        y = self.input_shape[2] - self.conv_kernel + 2 * self.conv_padding + 1
        z = self.input_shape[3] - self.conv_kernel + 2 * self.conv_padding + 1 

        # Calculate the effect of pooling
        if str(self.pooling_type).lower() != "none":
            x = x // self.pooling_kernel 
            y = y // self.pooling_kernel
            z = z // self.pooling_kernel 

        return (n_channels, x, y, z)

class dense(layer):
    """
    Dense layer implementation in eztorch4conv environment.
    It includes easy creation of dropout. It inherits the methods and
    attributes from the layer class
    """
    def __init__(self, input_shape=None, **kwargs):
        """
        Instanciates the class object. Parameters are to be introduced
        explictily as if they were keys in a dictionary. 

        Parameters
        ----------
        activation_function : torch activation function object
            Torch activation function object
            By default : torch.nn.ELU()
        dropout : float
            Proportion of neurons to be randomly dropout
            By default : 0
        input_shape : tuple
                Tuple with the dimensions of the input data with shape
                (n_channels, x, y, z)
        neurons : int
            Number of neurons that will compose the layer
        
        Returns
        -------
        dense : dense class instance

        Example 
        -------
        ez.layers.dense(neurons=1, dropout=0, activation_function=nn.Sigmoid())
        """
        # Check all input argumetns and set the default values if needed. 
        # Handle any errors that may arise.

        arguments = kwargs.keys()

        if 'neurons' not in arguments: raise KeyError('Please specify how many neurons comprise each layer')

        if 'activation_function' not in arguments: kwargs['activation_function'] = nn.ELU()

        if 'dropout' not in arguments: kwargs['dropout'] = 0

        super().__init__(input_shape, neurons=kwargs['neurons'], activation_function=kwargs['activation_function'],
                        dropout = kwargs['dropout'])
        
    def _create_main_layer(self):
        """
        Helper function to build_layer(). It creates the main dense layer, with
        the appropriate input and output shapes and appends it to the modules list.
        """
        # Create a dense layer with as many inputs as outputs the previous layer had

        self.in_channels = self.input_shape
        self.modules.append(nn.Linear(in_features=self.in_channels, out_features=self.neurons))

    def calculate_output_shape(self):
        """
        Calculates the output shape of the layer to facilitate the creation of 
        subsequent layers of the model.

        Returns
        -------
        output_shape : int
                    Integer with the shape of the output of the layer
        """
        return self.neurons

class flatten(layer):
    """
    Flatten implementation in eztorch4conv environment.
    It includes the calculation of output shape that is necessary
    for the automatic creation of all the layers in the model. It 
    inherits attributes and methods from the layer class.
    """
    def __init__(self):
        """       
        Instanciates the class object.

        Returns
        -------
        flatten : flatten class instance
        """
        self.input_shape = []

    def build_layer(self):
        """
        Creates a torch.nn.Flatten(1, -1) layer.

        Returns
        -------
        layer : torch.nn.Flatten(-1, 1)
            torch.nn.Flatten(-1, 1) to transform multidimensional
            tensors to a 1D tensor. 
        """
        return nn.Flatten(1, -1)

    def calculate_output_shape(self):
        """
        Calculates the output shape of the layer to facilitate the creation of 
        subsequent layers of the model.

        Returns
        -------
        output_shape : int
                    Integer with the shape of the output of the layer
        """
        return  self.input_shape[0] * self.input_shape[1] * self.input_shape[2] * self.input_shape[3]
