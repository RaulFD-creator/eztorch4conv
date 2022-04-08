"""
Architectures module within eztorch4conv library. It contains the basic classes necessary to construct
DCCN and MC-DCNN models with Pytorch in a simple and efficient way.
"""
import torch
import torch.nn as nn
import os
import copy

class Channel(nn.Module):
    def __init__(self):

        super(Channel, self).__init__()

        self.layers = nn.ModuleList()

    def add_layers(self, other):        
        for layer in other:
            if len(self.layers) == 0:
                self.input_shape = layer.input_shape

            if len(self.layers) != 0 or layer.input_shape is None:        
                layer.input_shape = self.prev_layer.calculate_output_shape()
            self.prev_layer = layer
            self.layers.append(layer.build_layer())  

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DCNN(nn.Module):
    """
    Class used to represent a DCNN (Deep Convolutional Neural Network) with Pytorch.
    It inherits from the nn.Module Pytorch class.
    
    Attributes
    ----------
    self.name : str
               Name of the model
                
    self.path : str
               Path where the model is to be stored
                
    self.params : dict
                 Dictionary where the training values will be recorded
                  
    self.layers : torch.nn.ModuleList
                 Special list where all the layers.layer objects
                 that comprise the model will be stored
                  
    self.error : torch.nn loss function
                Pytorch object that contains the loss function for the 
                optimisation
               
    self.optimizer : torch.nn optimizer algorithm
                    Pytorch object that contains the optimisation algorithm
                    that will be followed during training
    
    self.scheduler : torch.optim scheduler 
                    Pytorch object that contains a scheduler object
                    to dynamically change the learning rate
                     
    self.callbacks : list
                    list of eztorch4conv objects that control
                    the training process
                  
    Methods
    -------
    add_callbacks(other)
        Introduces a new callback object into self.callbacks
        
    add_layers(other)
        Introduces a new layer into the model
        
    add_scheduler(other)
        Introduces a scheduler to dynamically change
        the learning rate
        
    count_parameters()
        Returns the number of parameters the model will
        have to optimise
    
    define_loss(other)
        Defines what the loss function will be
    
    define_optimizer(other)
        Defines which optimisation algorithm the training
        will follow
        
    foward()
        Pytorch required method to define the order of layers
        within the model
        
    
    """
    def __init__(self, name, path='.'):

        super(DCNN, self).__init__()
        self.layers = nn.ModuleList()


        self.callbacks = []
        self.name = name
        self.path = os.path.join(path, self.name)
        self.float()
        self.params = {}
        try: 
            os.mkdir(self.path)

        except OSError: 
            raise Exception(f"Directory already existing: {self.path}\nSelect another name")

        with open(os.path.join(self.path, f'{self.name}_training.log'), "w") as of:
            of.write("Metric,Epoch,Mode,Training,Validation")
        with open(os.path.join(self.path, f"{self.name}.data"), "w") as of:
            of.write("Model\tEpoch\n")

    def add_callbacks(self, other):
        for callback in other:
            self.callbacks.append(callback)
    
    def add_layers(self, other):
        for layer in other:
            if len(self.layers) == 0:
                self.input_shape = layer.input_shape

            if len(self.layers) != 0 or layer.input_shape is None:        
                layer.input_shape = self.prev_layer.calculate_output_shape()
            self.prev_layer = layer
            self.layers.append(layer.build_layer())  
    
    def define_loss(self, other):
        self.error = other

    def define_optimizer(self, other):
        self.optimizer = other
    
    def add_scheduler(self, other):
        self.scheduler = other

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train_model(self, train_dataloader, validate_dataloader, len_training, 
                epochs, batch_size, metrics=["accuracy", "loss"]):

        self._init_training(metrics)
        self.float()

        for epoch in range(epochs):
            print()
            print("-"*11)
            print(f"| Epoch {epoch+1} |")
            print("-"*11)

            for mode in ['train', 'validate']:
                self.train() if mode == 'train' else self.eval()
                
                TP = 0
                TN = 0
                FP = 0
                FN = 0

                for i, (images, labels) in enumerate(train_dataloader if mode == 'train' else validate_dataloader):
                    # Load data and send to device
                    inputs = images.view(len(images),
                                                    self.input_shape[0],
                                                    self.input_shape[1],
                                                    self.input_shape[2],
                                                    self.input_shape[3]).to(self.device)

                    labels = labels.float().unsqueeze(1).to(self.device)
                    # Clear gradients
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(mode == 'train'):  
                        # Foward propagation if training, evaluation if not
                        outputs = self(inputs)

                        # Calculate loss
                        loss = self.error(outputs, labels)

                    if mode == 'train':
                        # Calculating gradients
                        loss.backward()
                        # Update parameters
                        self.optimizer.step()
                        # Update learning rate
                        try:
                            self.scheduler.step()
                        except:
                            pass
                    
                    TP_batch = 0
                    FN_batch = 0
                    TN_batch = 0
                    FP_batch = 0

                    for idx in range(len(labels)):
                        if outputs[idx] > 0.5 and labels[idx] == 1:
                            TP_batch += 1

                        elif outputs[idx] < 0.5 and labels[idx] == 1:
                            FN_batch += 1

                        elif outputs[idx] < 0.5 and labels[idx] == 0:
                            TN_batch += 1
                        
                        elif outputs[idx] > 0.5 and labels[idx] == 0:
                            FP_batch += 1

                    batch_acc = (TP_batch+TN_batch)/(TP_batch+TN_batch+FP_batch+FN_batch)

                    if mode == 'train':
                        print(f"Epoch {epoch+1}/{epochs}: Batch {i}/{len_training//batch_size} \tLoss: {loss.data}\t Accuracy: {batch_acc}")
                
                TP += TP_batch
                TN += TN_batch
                FP += FP_batch
                FN += FN_batch

                self._eval_performance(TP, TN, FP, FN, loss, epoch, mode)
                #self.check_callbacks()

        self._save_model(epoch, True)            

    def _save_model(self, epoch, final=False):
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
            of.write(f"{self.name.split('_')[0]}_{current_run}.pt\t {epoch}\n")
        
        if final:
            print()
            print("-"*21)
            print("| Stopping training |")
            print("-"*21)

    def _print_params(self, epoch):

        print()
        with open(os.path.join(self.path, f'{self.name}_training.log'), "a") as of:
            print(f'Epoch: {epoch+1}\tTraining\tValidation\n')
            for metric in self.metrics:
                of.write(f"{metric},{epoch+1},{self.params['train'][metric][epoch]},{self.params['validate'][metric][epoch]}")
                print(f"{metric}:\t{self.params['train'][metric][epoch]}\t{self.params['validate'][metric][epoch]}")
    
    def _check_callbacks(self):
        for callback in self.callbacks:
                callback.run()

    def _init_training(self, metrics):
        print(f"Training Model using device: {self.device}\n")

        available_metrics = ['accuracy', 'loss', 'sensitivity', 'precision', 'recall', 
                             'TP', 'TN', 'FP', 'FN', 'negative_predictive_value',
                             'f1', 'f2']
        self.params = {}

        if metrics == 'all':
            self.metrics = available_metrics
        for mode in ['train', 'validate']:
            self.params[mode] = {}
        
        for mode in ['train', 'validate']:
            for metric in available_metrics:
                self.params[mode][metric] = []

    def _eval_performance(self, TP, TN, FP, FN, loss, epoch, mode):

        # Performance metrics
        accuracy =  ((TN + TP) / (TP + TN + FP + FN))  
        try:
            precision = TP / (TP + FP) 
        except ZeroDivisionError:
            precision = 1
        try:
            negative_predictive_value = TN / (TN + FN)
        except ZeroDivisionError:
            negative_predictive_value = 1
        try:
            sensitivity = TP / (TP + FN)  # Same as recall
        except ZeroDivisionError:
            sensitivity = 1
        try:
            f1 = (2 * precision * sensitivity) / (precision + sensitivity)
        except ZeroDivisionError:
            f1 = 1
        try: 
            f2 = (1 + 2**2) * (2 * precision * sensitivity) / ((2**2) * precision + sensitivity)
        except ZeroDivisionError:
            f2 = 1

        self.params[mode]['accuracy'].append(accuracy)
        self.params[mode]['loss'].append(loss)
        self.params[mode]['TP'].append(TP)
        self.params[mode]['FP'].append(FP)
        self.params[mode]['TN'].append(TN)
        self.params[mode]['FN'].append(FN)
        self.params[mode]['precision'].append(precision)
        self.params[mode]['negative_predictive_value'].append(negative_predictive_value)
        self.params[mode]['sensitivity'].append(sensitivity)
        self.params[mode]['recall'].append(sensitivity)
        self.params[mode]['f1'].append(f1)
        self.params[mode]['f2'].append(f2)


        if mode == 'validate':
            self._print_params(epoch)
            if epoch % 10 == 0 and epoch != 0:
                self._save_model(epoch)
            elif self.params['validate']['accuracy'][-1] >= 0.7:
                self._save_model(epoch)   
   
class MCDCNN(DCNN):

    def __init__(self, name, path, n_channels, input_shape):

        super(DCNN, self).__init__()
        self.channels = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.n_channels = n_channels
        self.input_shape = input_shape
        self.callbacks = []
        self.name = name
        self.path = os.path.join(path, self.name)
        self.float()
        self.params = {}
        try: 
            os.mkdir(self.path)

        except OSError: 
            raise Exception(f"Directory already existing: {self.path}\nSelect another name")

        with open(os.path.join(self.path, f'{self.name}_training.log'), "w") as of:
            of.write("Metric,Epoch,Mode,Training,Validation")
        with open(os.path.join(self.path, f"{self.name}.data"), "w") as of:
            of.write("Model\tEpoch\n")
    
        for _ in range(self.n_channels):
            self.channels.extend([Channel()])
        
    def add_layers_to_channels(self, channels, layers):
        if channels == "all":
            for channel in self.channels:
                channel.add_layers(copy.deepcopy(layers))
        else:
            for channel in channels:
                channel.add_layers(copy.deepcopy(layers))

    def add_layers(self, other):

        for layer in other:
            if len(self.layers) == 0:
                layer.input_shape = 0
                for channel in self.channels:
                    layer.input_shape += channel.prev_layer.calculate_output_shape()

            if len(self.layers) != 0 or layer.input_shape is None:        
                layer.input_shape = self.prev_layer.calculate_output_shape()
                
            self.prev_layer = layer
            self.layers.append(layer.build_layer())  
   
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

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("Hello world")
