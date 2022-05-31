import torch.nn as nn

class dnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = []
    
    def forward(self, x):
        return self.layers(x)

class dcnn(nn.Module):
    def __init__(self) -> None:

        super().__init__()
        self.features = []
        self.classifier = []
        self.flatten = []

        message = "\nUsing DCNN architecture"
        print(message)
        print("-"*len(message))

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)

class mcdcnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.channels = nn.ModuleList()
        self.classifier = []

        message = "\nUsing MC-DCNN architecture"
        print(message)
        print("-"*len(message))

    def forward(self, x):
        outs = []
        for i, channel in enumerate(self.channels):
            out = channel(x[:,i,:,:,:].view(len(x), *channel.input_shape))
            outs.append(out)

        n = out.size()[-1] * len(outs)
        # concatenate channel outs
        x = torch.cat(outs,dim=0)
        x = x.view(-1,n)

        return self.classifier(x)

class Channel(nn.Module):
    """
    Class used to represent each of the Channels in a MC-DNN. Fundamentally,
    it works as the DNN class, but without all the functions focused on 
    training and performance recording. If multiplied by a positive integer
    'n', it returns a list with n deepcopies of itself.

    Attributes
    ----------
    layers : torch.nn.ModuleList
        Special list where all the layers.layer objects
        that comprise the model will be stored

    prev_layer : layer object
            Last layer object to be introduced in the ModuleList
            necessary for concatenating different layers, without
            having to specificate their input sizes.

    Methods
    -------
    add_layers(layers)
        Introduces a new layer into the model

    foward()
        Pytorch required method to define the order of layers
        within the model

    """
    def __init__(self, input_shape : tuple):
        """
        Creates an instance of the Channel class as a Pytorch Module.
        """
        super(Channel, self).__init__()

        self.layers = []
        self.input_shape = input_shape
    
    def __mul__(self, other):
        """
        Overrides operator '*' for Channel class. Now, it allows for the rapid creation
        of multiple channels.

        Parameters
        ----------
        other : int
            Number of identical channels to be created, has to be > 0
        
        Returns
        -------
        new_channels : list
                    List of 'other' number of deepcopies of original Channel
        """
        if isinstance(other, int) and other > 0:
            return [copy.deepcopy(self) for _ in range(other)]
        else:
            raise TypeError(f"Value: {other} is not an integer. Channels can only be multiplied by integers")

    def forward(self, x):
        """
        Pytorch required function that indicates how the information
        should flow through the different layers.

        Returns
        -------
        x : float
            Model prediction
        """
        return self.layers(x)
            