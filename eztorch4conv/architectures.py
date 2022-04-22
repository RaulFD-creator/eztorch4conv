# Copyright by Raúl Fernández Díaz

"""
Architectures module within eztorch4conv library. It contains the basic classes necessary to construct
DNN and MC-DNN models with Pytorch in a simple and efficient way. Classes are specifically designed
to create DCNN and MC-DCNN models, though they are general enough that they can be used for any kind of 
model.
"""
import eztorch4conv as ez
import torch
import torch.nn as nn
import os
import copy
import time

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
    def __init__(self):
        """
        Creates an instance of the Channel class as a Pytorch Module.
        """
        super(Channel, self).__init__()

        self.layers = nn.ModuleList()
    
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

    def add_layers(self, layers):  
        """
        Saves layer objects in the self.layers ModuleList.

        Parameters
        ----------
        layers : list or str
            Layer or list of layers to compose the model
        """
      
        if isinstance(layers, list):
            for layer in layers:
                if len(self.layers) == 0:
                    self.input_shape = layer.input_shape

                elif len(self.layers) != 0 or layer.input_shape is None:        
                    layer.input_shape = self.prev_layer.calculate_output_shape()
                self.prev_layer = layer
                self.layers.append(layer.build_layer())  

        else:
            if len(self.layers) == 0:
                    self.input_shape = layers.input_shape
            elif len(self.layers) != 0 or layers.input_shape is None:        
                layers.input_shape = self.prev_layer.calculate_output_shape()

            self.prev_layer = layers
            self.layers.append(layers.build_layer().to(self.device))  

    def forward(self, x):
        """
        Pytorch required function that indicates how the information
        should flow through the different layers.

        Returns
        -------
        x : float
            Model prediction
        """
        for layer in self.layers:
            x = layer(x)
        return x

class DNN(nn.Module):
    """
    Class used to represent a DNN (Deep Neural Network) model with Pytorch.
    It inherits from the torch.nn.Module Pytorch class.
    
    Attributes
    ----------
    available_metrics : list
                List of supported metrics

    callbacks : list
            List of eztorch4conv objects that control
            the training process

    checkpoints : list
            List of checkpoints to store all checkpoints
    
    error : torch.nn loss function
        Pytorch object that contains the loss function for the 
        optimisation

    layers : torch.nn.ModuleList
        Special list where all the layers.layer objects
        that comprise the model will be stored
    
    name : str
        Name of the model

    optimizer : torch.nn optimizer algorithm
            Pytorch object that contains the optimisation algorithm
            that will be followed during training
                
    params : dict
            Dictionary where the training values will be recorded

    path : str
        Path where the model is to be stored

    save_files : bool
            Flag that indicates whether the history training
            and model checkpoints should be stored

    stop_training : bool
                Flag that indicates whether the training
                should be stopped
                   
    scheduler : torch.optim scheduler 
            Pytorch object that contains a scheduler object
            to dynamically change the learning rate
                          
    Methods
    -------
    add_callbacks(callbacks)
        Introduces a new callback object into self.callbacks
        
    add_layers(layers)
        Introduces a new layer into the model
        
    compile(optimizer, loss_function, scheduler, device, device_ID)
        Defines the optimisation algorithm that the modil will
        follow during training; the loss_function it will
        try to minimize and it may introduces a scheduler to dynamically 
        change the learning rate. It also prepares the model
        to be run in GPU.
        
    count_parameters()
        Returns the number of parameters the model will
        have to optimise
        
    foward()
        Pytorch required method to define the order of layers
        within the model
        
    train_model(train_dataloader, validate_dataloader, len_training, 
                epochs, batch_size, metrics)
        Trains and validates the model      
    """
    def __init__(self, name, path='.', save_files=True):
        """
        Create an instance of the DNN class as a Pytorch Module and 
        creates the appropriate directories and files required
        for storing the output information.

        Parameters
        ---------
        name : str
            Name of the model, will be used to store the different
            output files

        path : str
            Directory where the output files, including trained models,
            will be stored
        """
        super(DNN, self).__init__()

        # Initialise attributes
        self.available_metrics = ['accuracy', 'loss', 'sensitivity', 'specificity', 'precision', 'recall',
                        'TP', 'TN', 'FP', 'FN', 'negative_predictive_value',
                        'f1', 'f2']
        self.callbacks = []
        self.checkpoints = []
        self.layers = nn.ModuleList()
        self.name = name
        self.history = {}
        self.path = os.path.join(path, self.name)
        self.save_files = save_files

        # Initialise model
        self.float()

        # Create appropiate directories and files to 
        # store the trained models and the training data
        if save_files:
            try: 
                os.mkdir(self.path)

            except OSError: 
                raise OSError(f"Directory already exists: {self.path}\nPlease select another name")

            with open(os.path.join(self.path, f'{self.name}_training.log'), "w") as of:
                of.write("Metric,Epoch,Mode,")
            with open(os.path.join(self.path, f"{self.name}.data"), "w") as of:
                of.write("Model\tEpoch\n")

    def add_callbacks(self, callbacks):
        """
        Saves callback objects in the self.callbacks attribute.
        Callback objects are evaluated at the end of an epoch
        and may regulate how the model behaves. Some examples
        are early stopping or saving checkpoints when the model
        obbtains an accuracy score greater than a threshold.

        Parameters
        ----------
        callbacks : list or str
            Callback or list of callbacks to regulate
            in the model
        """
        if isinstance(callbacks, list):
            for callback in callbacks:
                    self.add_callbacks(callback)
        elif isinstance(callbacks, ez.callbacks.Callback):
            self.callbacks.append(callbacks)
        else:
            raise TypeError(f"Object {callbacks} is neither a callback nor a list of callbacks.")
                    
    def add_layers(self, layers):
        """
        Saves layer objects in the self.layers ModuleList.

        Parameters
        ----------
        layers : list or str
            Layer or list of layers to compose the model
        """
        if isinstance(layers, list):
            for layer in layers:
                if len(self.layers) == 0:
                    self.input_shape = layer.input_shape

                elif len(self.layers) != 0 or layer.input_shape is None:        
                    layer.input_shape = self.prev_layer.calculate_output_shape()
                self.prev_layer = layer
                self.layers.append(layer.build_layer())  

        else:
            if len(self.layers) == 0:
                    self.input_shape = layers.input_shape
            elif len(self.layers) != 0 or layers.input_shape is None:        
                layers.input_shape = self.prev_layer.calculate_output_shape()

            self.prev_layer = layers
            self.layers.append(layers.build_layer())  
    
    def compile(self, optimizer, loss_function, scheduler=None, device='cpu', device_ID=0):
        """
        Defines the optimizer, loss function, and possible learning
        rate scheduler that the model will use for training. It also
        prepares model to be run in GPU.

        Parameters
        ----------
        optimizer : Pytorch optim object
                Pytorch optimizer

        loss_function : Pytorch loss object
                    Pytorch loss function

        schduler : Pytorch optim object
                Pytorch scheduler

        device : str
            Device in which the model should be trained
            By default: 'cpu'

        device_ID : int
                Device ID when working with GPUs
                By default : 0
        """
        self.error = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler

        # If GPU is to be used, prepare the system for optimal performance
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(device_ID)

        # Send model to the appropriate device
        self.to(device)
        self.device = device

    def count_parameters(self):
        """
        Count the number of parameters the model will have to train.
        It is recommended there are no more than 10 times less than
        the number of features the model will be trained with.

        Returns
        -------
        num_parameters : int
                Number of parameters the model will train
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        """
        Pytorch required function that indicates how the information
        should flow through the different layers.

        Returns
        -------
        x : float
            Model prediction
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def train_model(self, train_dataloader, validate_dataloader, len_training, 
                epochs, batch_size, metrics=["accuracy", "loss"]):
        """
        Trains the model.

        Parameters
        ----------
        train_dataloader : Pytorch DataLoader object
                    Pytorch DataLoader object that contains the training data

        validate_dataloader : Pytorch DataLoader object
                        Pytorch DataLoader object that contains the validation data

        len_training : int
                    Number of entries in the training data

        epochs : int
            Number of times the model will iterate through all the training data

        batch_size : int
                Number of entries the model will consider for computing each step

        metrics : list
                List of metrics that will be computed and stored to follow
                the training process
                By default: ["accuracy", "loss"]
        """
        # Initialise training and create appropriate dependencies
        # record the training process
        self._init_training(metrics)

        # Training loop
        for epoch in range(epochs):
            print()
            print("-"*11)
            print(f"| Epoch {epoch+1} |")
            print("-"*11)

            # Loop through 2 modes: Training and validation
            for mode in ['train', 'validate']:

                # Set the model to the appropriate mode
                self.train() if mode == 'train' else self.eval()
                
                # Initialise model performance counters
                TP = 0
                TN = 0
                FP = 0
                FN = 0

                # Loop through the items in the DataLoaders
                for i, (images, labels) in enumerate(train_dataloader if mode == 'train' else validate_dataloader):
                    start = time.time()
                    # Load data and send to device
                    inputs = images.view(
                                        len(images),
                                        self.input_shape[0],
                                        self.input_shape[1],
                                        self.input_shape[2],
                                        self.input_shape[3]
                                        ).to(self.device)
                    labels = labels.float().unsqueeze(1).to(self.device)
                   
                    # Clear gradients
                    self.optimizer.zero_grad()

                    # Prepare gradients if in training mode
                    with torch.set_grad_enabled(mode == 'train'):  

                        # If training mode, foward propagation; 
                        # if validation  mode, evaluate samples
                        outputs = self(inputs)

                        # Calculate loss
                        loss = self.error(outputs, labels)

                    if mode == 'train':

                        # Calculating gradients
                        loss.backward()
                        # Update parameters
                        self.optimizer.step()
                        # Update learning rate
                        if self.scheduler is not None: self.scheduler.step()
                    
                    # Initialise batch model performance counters
                    TP_batch = 0
                    FN_batch = 0
                    TN_batch = 0
                    FP_batch = 0

                    # Calculate batch model performance
                    for idx in range(len(labels)):
                        if outputs[idx] > 0.5 and labels[idx] == 1: TP_batch += 1
                        elif outputs[idx] < 0.5 and labels[idx] == 1: FN_batch += 1
                        elif outputs[idx] < 0.5 and labels[idx] == 0: TN_batch += 1
                        elif outputs[idx] > 0.5 and labels[idx] == 0: FP_batch += 1

                    # Calculate model batch accuracy
                    batch_acc = (TP_batch+TN_batch)/(TP_batch+TN_batch+FP_batch+FN_batch)

                    end = time.time()
                    if mode == 'train':
                        print(f"Epoch {epoch+1}/{epochs}: Batch {i}/{len_training//batch_size} Loss: {loss.data} Accuracy: {batch_acc} Time: {end-start} s")
                
                    # Compute basic model performance
                    TP += TP_batch
                    TN += TN_batch
                    FP += FP_batch
                    FN += FN_batch

                # Compute advanced model performance metrics
                self._eval_performance(TP, TN, FP, FN, loss, epoch, mode)

                # Check callbacks
                if mode == 'validate': self._check_callbacks(epoch)
            if self.stop_training: break

        if not self.stop_training: self._save_model(epoch, True)            
        return self.checkpoints, self.history

    def _check_callbacks(self, epoch):
        """
        Helper function to train_model(). Checks all callbacks and if necessary
        executes whatever function they contain.
        """
        for callback in self.callbacks:
                callback.run(epoch)

    def _eval_performance(self, TP, TN, FP, FN, loss, epoch, mode):
        """
        Helper function to train_model(). Computes advanced model 
        performance metrics and records those the user has indicated
        in the self.history dictionary.
        """
        # Performance metrics
        accuracy =  ((TN + TP) / (TP + TN + FP + FN))  

        try: precision = TP / (TP + FP) 
        except ZeroDivisionError: precision = 0

        try: negative_predictive_value = TN / (TN + FN)
        except ZeroDivisionError: negative_predictive_value = 0

        try: sensitivity = TP / (TP + FN)  # Same as recall
        except ZeroDivisionError: sensitivity = 0

        try: specificity = TN / (TN + FP)
        except ZeroDivisionError: specificity = 0

        try: f1 = (2 * precision * sensitivity) / (precision + sensitivity)
        except ZeroDivisionError: f1 = 0

        try:  f2 = (1 + 2**2) * (2 * precision * sensitivity) / ((2**2) * precision + sensitivity)
        except ZeroDivisionError: f2 = 0

        self.history[mode]['accuracy'].append(accuracy)
        self.history[mode]['loss'].append(loss)
        self.history[mode]['TP'].append(TP)
        self.history[mode]['FP'].append(FP)
        self.history[mode]['TN'].append(TN)
        self.history[mode]['FN'].append(FN)
        self.history[mode]['precision'].append(precision)
        self.history[mode]['negative_predictive_value'].append(negative_predictive_value)
        self.history[mode]['sensitivity'].append(sensitivity)
        self.history[mode]['specificity'].append(specificity)
        self.history[mode]['recall'].append(sensitivity)
        self.history[mode]['f1'].append(f1)
        self.history[mode]['f2'].append(f2)
        
        if mode == 'validate':
            self._print_history(epoch, mode)

    def _init_training(self, metrics):
        """
        Helper function to train_model(). Initialises the self.history dictionary
        to be able to record training evolution.
        """
        print(f"Training Model using device: {self.device}\n")
        self.stop_training = False

        if metrics == 'all':
            self.metrics = []

            for metric in self.available_metrics:
                if metric != 'recall':
                    self.metrics.append(metric)
        
        if isinstance(metrics, list):
            for metric in metrics:
                self.metrics.append(metric) if metric in self.available_metrics else print(f"Warning: Metric: {metric} is not supported.\nPlease one of the supported metrics: {self.available_metrics}")

        for mode in ['train', 'validate']:
            self.history[mode] = {}
        
        for mode in ['train', 'validate']:
            for metric in self.available_metrics:
                self.history[mode][metric] = []

        if self.save_files:
            k = 2
            for metric in self.metrics:
                with open(os.path.join(self.path, f'{self.name}_training.log'), "a") as of:
                    of.write(f"{metric},") if k < len(self.available_metrics) else of.write(f"{metric}\n")
                    k += 1


    
    def _print_history(self, epoch : int, mode : str):
        """
        Helper function to train_model(). Outputs and stores the performance values for the predefined metrics
        at the end of each epoch.
        """
        print()
        num_metrics = len(self.available_metrics)
        if self.save_files:
            with open(os.path.join(self.path, f'{self.name}_training.log'), "a") as of:
                print(f'Epoch: {epoch+1}\tTraining\tValidation\n')

                for submode in ['train', 'validate']:
                    of.write(f"{epoch+1},{submode}")
                    k = 2
                    for metric in self.metrics:
                        of.write(f",{self.history[submode][metric][epoch]}") if k < num_metrics else of.write(f",{self.history[submode][metric][epoch]}\n") 
                        if submode == 'train':
                            print(f"{metric}:\t{self.history['train'][metric][epoch]}\t{self.history['validate'][metric][epoch]}")
                        k += 1
        else:
            print(f'Epoch: {epoch+1}\tTraining\tValidation\n')
            for metric in self.metrics:
                if metric in self.available_metrics:
                    print(f"{metric}:\t{self.history['train'][metric][epoch]}\t{self.history['validate'][metric][epoch]}")
                else:
                    print(f"Warning: Metric: {metric} is not supported.\nAvailable metrics: {self.available_metrics}")
            
    def _save_model(self, epoch, final=False):
        """
        Helper method to train_model(). It saves an instance of the model, and records
        to which epoch it corresponds to facilitate checking what its performance values
        were.
        """
        if self.save_files:
            previous_runs = -1
            for file in os.listdir(self.path):
                try:
                    if file.split('_')[0] == self.name.split('_')[0] and file.split('.')[1] == 'pt':
                        previous_runs += 1
                except IndexError:
                    continue

            current_run = previous_runs + 1
            torch.save(self, os.path.join(self.path, f"{self.name.split('_')[0]}_{current_run}.pt"))
            with open(os.path.join(self.path, f"{self.name}.data"), "a") as of:
                of.write(f"{self.name.split('_')[0]}_{current_run}.pt\t{epoch}\n")
            
        else:
            self.checkpoints.append((epoch, copy.copy(self)))

        if final:
            print()
            print("-"*21)
            print("| Stopping training |")
            print("-"*21)
            self.stop_training = True

class MCDNN(DNN):
    """
    Class used to represent a MC-DNN (Multi-Channel Deep Neural Network) 
    with Pytorch. It inherits from the DNN class which, in turn, inherited from the
    torch.nn.Module class. Methods add_layers and forward work similarly to those
    from its parent class but have been adapted to idiosyncrasies inherent to 
    the multi-channel processing.

    Attributes
    ----------
    channels : torch.nn.ModuleList()
            Special list that contains all the channels that compose
            the MC-DNN
    
    n_channels : int
            Number of channels it has to be a positive integer

    input_shape : tuple
            Shape of input files to the MC-DNN. (n_channels, x, y, z)    

    Methods
    -------
    add_channels(channels) 
        Introduce a new channel or list of channels into the self.channels ModuleList

    add_layers_to_channels(channels, layers)   
        Introduce a layer or list of layers to the specified channels. 

    """
    def __init__(self, name, path, input_shape, n_channels=0, save_files=False):

        super(DNN, self).__init__()
        self.available_metrics = ['accuracy', 'loss', 'sensitivity', 'specificity', 'precision', 'recall',
                'TP', 'TN', 'FP', 'FN', 'negative_predictive_value',
                'f1', 'f2']
        self.channels = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.n_channels = n_channels
        self.input_shape = input_shape
        self.callbacks = []
        self.checkpoints = []
        self.name = name
        self.path = os.path.join(path, self.name)
        self.history = {}
        self.save_files = save_files

        if self.save_files:
            try: 
                os.mkdir(self.path)

            except OSError: 
                raise OSError(f"Directory already existing: {self.path}\nPlease select another name")

            with open(os.path.join(self.path, f'{self.name}_training.log'), "w") as of:
                of.write("Metric,Epoch,Mode,Training,Validation\n")
            with open(os.path.join(self.path, f"{self.name}.data"), "w") as of:
                of.write("Model\tEpoch\n")

        if n_channels > 0:
            self.add_channels(Channel() * self.n_channels)
        
    def add_channels(self, channels):
        if isinstance(channels,list):
            for channel in channels:
                self.add_channels(channel)
        elif isinstance(channels, Channel):
            self.channels.append(channels)
            self.n_channels += 1
        else:
            raise TypeError(f"Object {channels} is neither a channel nor a list of channels")

    def add_layers(self, layers):
        for layer in layers:
            if len(self.layers) == 0:
                layer.input_shape = 0
                for channel in self.channels:
                    layer.input_shape += channel.prev_layer.calculate_output_shape()

            if len(self.layers) != 0 or layer.input_shape is None:        
                layer.input_shape = self.prev_layer.calculate_output_shape()
                
            self.prev_layer = layer
            self.layers.append(layer.build_layer())  
          
    def add_layers_to_channels(self, channels, layers):
        if channels == "all":
            for channel in self.channels:
                if isinstance(layers, list):
                    channel.add_layers(copy.deepcopy(layers))
                else:
                    channel.add_layers(copy.deepcopy(layers))
        else:
            for channel in channels:
                if isinstance(layers, list):
                    channel.add_layers(copy.deepcopy(layers))
                else:
                    channel.add_layers(copy.deepcopy(layers))
                    
    def forward(self, x):
        outs = []
        for i, channel in enumerate(self.channels):
            out = channel(x[:,i,:,:,:].view(len(x),1,16,16,16))
            outs.append(out)

        n = out.size()[-1] * len(outs)
        # concatenate channel outs
        x = torch.cat(outs,dim=0)
        x = x.view(-1,n)

        for layer in self.layers:
            x = layer(x)
        return x

class DCNN(DNN):
    pass

class MCDCNN(MCDNN):
    pass