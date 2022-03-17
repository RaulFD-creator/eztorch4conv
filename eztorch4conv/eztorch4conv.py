"""
Docustring
"""
import sys
import torch
import torch.nn as nn
import pandas as pd
import os
import random
from torch.utils.data import Dataset
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
                        "device":
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

class DCNN(nn.Module):
    def __init__(self):

        super(DCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.dense_layers = nn.ModuleList()

        self.count = 0
        self.loss_list = []
        self.iteration_list = []
        self.accuracy_list = []
        self.loss_val_list = []
        self.iteration_list = []
        self.accuracy_val_list = []

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = "cpu"
        self.float()
        #self.to(self.device)
    
    def add_conv_layer(self, other):
        self.conv_layers.append(other)
    
    def add_dense_layer(self, other):
        self.dense_layers.append(other)
    
    def add_loss(self, other):
        self.error = other

    def add_optimizer(self, other):
        self.optimizer = other

    def Flatten(self, x):
        return x.view(x.size(0), -1)
    
    def forward(self, x):
        
        for layer in self.conv_layers:
            x = layer(x)

        x = self.Flatten(x)

        for layer in self.dense_layers:
            x = layer(x)
        return x

    def save_model(self, accuracy_val):
        current_number = 0
        for file in os.listdir("./Models"):
            try:
                pointer = file.split("_")[1]
                pointer = int(pointer.split(".")[0])
            except:
                pointer = 0
                pass
            if pointer > current_number:
                current_number = pointer
        new_number = current_number + 1
        with open("./Models/log.txt", "a") as fo:
            fo.write(f"model_{new_number};{accuracy_val}\n")
            fo.write(f"Input_parameters: {self.input_parameters}")
        print(f"Saving model as: ./Models/model_{new_number}.pt")
        torch.save(self, f"./Models/model_{new_number}.pt")
        print("Model saved.")

    def train(self, train_dataloader, validate_dataloader, len_training, 
                epochs, batch_size, model_name, input_parameters):
        self.input_parameters = input_parameters
        print(f"Training Model using device: {self.device}")
        with open(f"./Training_logs/dcnn_{model_name}.txt","a") as fo:
            fo.write(f"Input parameters: {input_parameters}\n")


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
                os.system(f"touch ./Training_logs/dcnn_{model_name}.txt")
                with open(f"./Training_logs/dcnn_{model_name}.txt","a") as fo:
                    fo.write(f"Epoch {epoch}/{self.num_epochs}: Batch {i}/{len_training//self.batch_size} \tLoss: {loss.data}\n")
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in validate_dataloader:
                
                test = torch.autograd.Variable(images.view(len(images),6,16,16,16))
                # Forward propagation
                outputs = self(test.float().to(self.device))
                # Get predictions from the maximum value
                for idx in range(len(labels)):
                    if outputs[idx] > 0.5 and labels[idx] == 1:
                        correct += 1
                    elif outputs[idx] < 0.5 and labels[idx] == 0:
                        correct += 1
                
                # Total number of labels
                total += len(labels)

                accuracy = 100 * correct / float(total)
                
                # store loss and iteration
                self.loss_list.append(loss.data)
                self.iteration_list.append(self.count)
                self.accuracy_list.append(accuracy)

            # Print Loss
            print('Loss training: {}  Validation accuracy: {} %'
                .format(loss.data, accuracy))
            with open(f"./Training_logs/dcnn_{model_name}.txt","a") as fo:
                    fo.write(f"Loss training: {loss.data} \tValidation accuracy: {accuracy}\n")
            
            if accuracy > 80 and self.count > 10:
                # Callback
                self.save_model(accuracy)
                
        self.save_model(accuracy)

class Channel(nn.Module):
    def __init__(self):

        super(Channel, self).__init__()

        self.conv_layers = nn.ModuleList()

    def add_conv_layer(self, other):
        self.conv_layers.append(other)

    def forward(self, x):
        for layer in self.conv_layers:
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
        self.dense_layers = nn.ModuleList()

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = "cpu"
        self.float()
        #self.to(self.device)

        for _ in range(self.n_channels):
            self.channels.extend([Channel()])
        
        for channel in self.channels:
            channel.add_conv_layer(conv3d(in_channels=1, out_channels=48, conv_kernel=5, 
                        pooling_type='max', dropout=0.25).build_layer())
            channel.add_conv_layer(conv3d(in_channels=48, out_channels=64, conv_kernel=5, conv_padding=1,
                            pooling_type='max', dropout=0.25).build_layer())
            channel.add_conv_layer(conv3d(in_channels=64, out_channels=96, conv_kernel=3, conv_padding=1,
                        pooling_type='max', dropout=0.25).build_layer())
    

    def save_model(self, accuracy_val):
        current_number = 0
        for file in os.listdir("./MC_Models"):
            try:
                pointer = file.split("_")[1]
                pointer = int(pointer.split(".")[0])
            except:
                pointer = 0
                pass
            if pointer > current_number:
                current_number = pointer
        new_number = current_number + 1
        with open("./MC_Models/log.txt", "a") as fo:
            fo.write(f"model_{new_number};{accuracy_val}\n")
            fo.write(f"Input_parameters: {self.input_parameters}")
        print(f"Saving model as: ./MC_Models/model_{new_number}.pt")
        torch.save(self, f"./MC_Models/model_{new_number}.pt")
        print("Model saved.")
    
    
    def forward(self, x):
        outs = []
        for i, channel in enumerate(self.channels):

            out = channel(x[:,i,:,:,:].view(len(x),1,16,16,16))
            out = self.Flatten(out)
            outs.append(out)

        n = out.size()[-1] * len(outs)
        # concatenate channel outs
        x = torch.cat(outs,dim=0)
        x = x.view(-1,n)

        for layer in self.dense_layers:
            x = layer(x)

        return x





if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
