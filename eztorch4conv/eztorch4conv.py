"""
Docustring
"""
import torch
import torch.nn as nn
import os
from .callbacks import *

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

class DCNN(nn.Module):
    def __init__(self, name, path='./'):

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
        self.params = {}

    def add_callback(self, other):
        self.callbacks.append(other)
    
    def add_layer(self, other):
        self.layers.append(other.build_layer())
    
    def define_loss(self, other):
        self.error = other

    def define_optimizer(self, other):
        self.optimizer = other

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        return x

    def save_model(self, final=False):

        torch.save(self, self.save + ".pt")
        os.system(f"touch {self.save + '.log'}")
        with open(self.save + ".log", "a") as fo:
            fo.write(f"{self.params}")
        
        if final:
            os.system(f"touch {self.save + '.data'}")
            with open(self.save + ".data", "a") as fo:
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

            # Calculate metrics         
            correct = 0
            incorrect = 0
            total = 0
            TP = 0
            FN = 0
            TN = 0
            FP = 0

            # Iterate through validation dataset
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

class MCDCNN(DCNN):

    def __init__(self, n_channels, name, path):

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
        self.name = name
        self.save = path
        self.params = {}

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
