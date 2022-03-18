"""
Docustring
"""
import sys
import torch
import torch.nn as nn
from torch.nn import Flatten as flatten
import os
import json
import sys

def parse_inputs(path_to_arguments):
    """
    Function to parse the file with the hyper-parameters of the network.

    Parameters
    ----------
    path_to_arguments : string
                        Contains the path to a '.json' that will describe the hyper-parameters 
                        of the network
    
    Returns
    -------
    input_parameters : dictionary with some of the hyper-parameters
                        "model_name": Name of the model (will be used to save the information),
                        "batch_size": Batch size,
                        "num_epochs": Number of epochs of training, 
                        "learning_rate": Learning Rate,
                        "weight_decay": Weight decay for L2 Regularization,
                        "num_classes": Number of different categories that the data is separated into, 
                        "num_channels": Number of channels used,
                        "dataset_proportion": ,
                        "device": CPU ('cpu') or GPU ('cuda'). By default, it will check if there is GPU
                                    available and use it; if not available, will use CPU.
    """
    path_to_arguments = sys.argv[1]

    with open(path_to_arguments, 'r') as j:
        return json.loads(j.read())

class conv3d():
    def __init__(self, in_channels, out_channels, conv_kernel, conv_padding=1, 
                pooling_type='max', pooling_kernel=2, dropout=0.25):

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

class early_stop():

    def __init__(self, metric, target):
        self.metric = metric
        self.target = target

    def check_condition(self, params):

        if self.metric != 'loss' and params[self.metric] >= self.target:
            return True

        elif self.metric == 'loss' and params[self.metric] <= self.target:
            return True

        else:
            return False

    def __str__(self):
        return 'early_stop'

class checkpoint(early_stop):

    def __str__(self):
        return 'checkpoint'


class DCNN(nn.Module):
    def __init__(self, name, path):

        super(DCNN, self).__init__()
        self.layers = nn.ModuleList()

        self.name = name
        self.count = 0
        self.loss_list = []
        self.iteration_list = []
        self.accuracy_list = []
        self.loss_val_list = []
        self.iteration_list = []
        self.accuracy_val_list = []
        self.callbacks = []
        self.save = os.path.join(path, name)
        self.float()
    
    def add_callback(self, other):
        self.callbacks.append(other)
    
    def add_layer(self, other):
        if other is not flatten:
            self.layers.append(other.build_layer())
        else:
            self.layers.append(other)
    
    def define_loss(self, other):
        self.error = other

    def define_optimizer(self, other):
        self.optimizer = other

    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)
        return x

    def save_model(self, final=False):
        torch.save(self, os.path.join(self.save, ".pt"))
        os.system(f"touch {os.path.join(self.save, '.log')}")
        with open(os.path.join(self.save, ".log"), "a") as fo:
            fo.write(self.params)
        
        if final:
            os.system(f"touch {os.path.join(self.save, '.data')}")
            with open(os.path.join(self.save, ".data"), "a") as fo:
                for metric in self.metrics:
                    fo.write(f"{metric},\t")
                fo.write(f"\n")

                for i in range(len(self.params['loss'])):
                    for metric in self.metrics:
                        fo.write(f"{self.params[metric][i]},\t")
                    fo.write(f"\n")

    
    def check_callbacks(self):
        for callback in self.callbacks:
                if callback is early_stop and callback.check_condition(self.params):
                    print(f"Stopping training and saving model because taget ({callback.metric}) has been achieved ({self.params[callback.metric]}/{callback.taget})")
                    self.save_model(True)
                    break
                if callback is checkpoint and callback.check_conditions(self.params):
                    print(f"Saving model because target {callback.metric} has been achieved ({self.params[callback.metric]}/{callback.taget})")
                    self.save_model()


    def train(self, train_dataloader, validate_dataloader, len_training, 
                epochs, batch_size, metrics=["accuracy", "loss"]):
        self.params = {}
        self.metrics = metrics
        for metric in self.metrics:
            self.params[metric] = []
        print(f"Training Model using device: {self.device}")

        self.num_epochs = epochs
        self.batch_size = batch_size
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(train_dataloader):
                
                train = torch.autograd.Variable(images.view(len(images),6,16,16,16))
                labels = torch.autograd.Variable(labels)
                # Clear gradients
                self.optimizer.zero_grad()
                # Forward propagation
                outputs = self(train.float().to(self.device))
                # Calculate loss
                loss = self.error(outputs, labels.float().unsqueeze(1).to(self.device))
                # Calculating gradients
                loss.backward()
                # Update parameters
                self.optimizer.step()
                
                self.count += 1

                print(f"Epoch {epoch}/{self.num_epochs}: Batch {i}/{len_training//self.batch_size} \tLoss: {loss.data}")

            # Calculate Accuracy         
            correct = 0
            incorrect = 0
            total = 0
            TP = 0
            FN = 0
            TN = 0
            FP = 0

            # Iterate through test dataset
            for images, labels in validate_dataloader:
                
                test = torch.autograd.Variable(images.view(len(images),6,16,16,16))

                # Forward propagation
                outputs = self(test.float().to(self.device))

                # Get predictions from the maximum value
                for idx in range(len(labels)):
                    if outputs[idx] > 0.5 and labels[idx] == 1:
                        correct += 1
                        TP += 1

                    elif outputs[idx] < 0.5 and labels[idx] == 1:
                        incorrect += 1
                        FN += 1

                    elif outputs[idx] < 0.5 and labels[idx] == 0:
                        correct += 1
                        TN += 1
                    
                    elif outputs[idx] > 0.5 and labels[idx] == 0:
                        incorrect += 1
                        FP += 1

                # Total number of labels
                total = len(labels)
                accuracy = 100 * correct / float(total)
                
                # store loss and iteration
                self.loss_list.append(loss.data)
                self.iteration_list.append(self.count)
                self.accuracy_list.append(accuracy)
                self.params['accuracy'].append(accuracy)
                self.params['loss'].append(loss)
                try:
                    self.params['TP'].append(TP)
                except KeyError:
                    continue

                try:
                    self.params['FP'].append(FP)
                except KeyError:
                    continue

                try:
                    self.params['TN'].append(TN)
                except KeyError:
                    continue

                try:
                    self.params['FN'].append(FN)
                except KeyError:
                    continue

            # Print Loss
            print(f"{self.params}")
            self.check_callbacks()

            
        self.save_model(True)
    
class Channel(nn.Module):
    def __init__(self):

        super(Channel, self).__init__()

        self.layers = nn.ModuleList()

    def add_layer(self, other):
        self.layers.append(other.build_layer())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MCDCNN(DCNN):

    def __init__(self, n_channels):

        super(DCNN, self).__init__()

        self.count = 0
        self.loss_list = []
        self.iteration_list = []
        self.accuracy_list = []
        self.loss_val_list = []
        self.iteration_list = []
        self.accuracy_val_list = []
        self.channels = nn.ModuleList()
        self.n_channels = n_channels
        self.layers = nn.ModuleList()

        self.float()

        for _ in range(self.n_channels):
            self.channels.extend([Channel()])
    
    def add_layer_to_channels(self, channels, layer):
        if channels == "all":
            for channel in self.channels:
                channel.add_layer(layer)
        else:
            for channel in channels:
                channel.add_layer(layer)
    
    def add_layer(self, layer):
        self.layers.append(layer.build_layer())
    

    def save_model(self, filename):
        torch.save(self, filename)
    
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
