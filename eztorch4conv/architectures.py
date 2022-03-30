"""
Docustring
"""
import torch
import torch.nn as nn
import os
import sys

class Channel(nn.Module):
    def __init__(self):

        super(Channel, self).__init__()

        self.layers = nn.ModuleList()

    def add_layer(self, other):
        if len(self.layers) != 0:        
            other.input_shape = self.prev_layer.calculate_output_shape()
        self.prev_layer = other
        self.layers.append(other.build_layer())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DCNN(nn.Module):
    def __init__(self, name, path='./'):

        super(DCNN, self).__init__()
        self.layers = nn.ModuleList()

        try: 
            os.mkdir(os.path.join(path, name))
            self.name = name
        
        except OSError: 
            count = 1
            for file in os.listdir(path):
                if file.split("_")[0] == name.split("_")[0]:
                    print(file, name)
                    count += 1
            new_name = name.split("_")[0] + '_' + str(count)
            self.name = new_name
            print(f"Directory already exists. Creating new directory: {os.path.join(path, new_name)}")
            os.mkdir(os.path.join(path, new_name))

        self.count = 0
        self.callbacks = []
        self.path = os.path.join(path, self.name)
        self.name = name
        self.float()
        self.params = {}

    def add_callbacks(self, other):
        for callback in other:
            self.callbacks.append(callback)
    
    def add_layers(self, other):
        for layer in other:
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

    def save_model(self, final=False):
        previous_runs = -1
        for file in os.listdir(self.path):
            if file.split('.')[0] == self.name and file.split('.')[1] == 'pt':
                previous_runs += 1
        current_run = previous_runs + 1
        torch.save(self, os.path.join(self.path, f'{current_run}.pt'))
        os.system(f"touch {os.path.join(self.path, f'{current_run}.data')}")
        with open(os.path.join(self.path, f'{self.name}.log'), "a") as fo:
            fo.write(f"{self.params}")
        
        if final:
            os.system(f"touch {os.path.join(self.path, f'{self.name}.data')}")
            with open(os.path.join(self.path, f'{self.name}.data'), "a") as fo:
                for metric in self.metrics:
                    fo.write(f"{metric}\t")
                fo.write(f"\n")

                for i in range(len(self.params['accuracy'])):
                    for metric in self.metrics:
                        fo.write(f"{self.params[metric][i]},\t")
                    fo.write(f"\n")
            print("Stopping training ")
            sys.exit(1)

    def print_params(self):
        print()
        os.system(f"touch {os.path.join(self.path, f'{self.name}_training.log')}")

        with open(os.path.join(self.path, f'{self.name}_training.log'), "a") as of:
            for metric in self.metrics:
                of.write(f"{self.params[metric][-1]},\t")
                print(f"{metric}: {self.params[metric][-1]}") 
            print() 

    
    def check_callbacks(self):
        for callback in self.callbacks:
                callback.run()


    def train(self, train_dataloader, validate_dataloader, len_training, 
                epochs, batch_size, metrics=["accuracy", "loss"]):

        print(f"Training Model using device: {self.device}")

        available_metrics = ['accuracy', 'loss', 'sensitivity', 'precision', 'recall', 
                            'specificity', 'TP', 'TN', 'FP', 'FN', 'negative_predictive_value',
                             'f1', 'f2']
        self.params = {}
        self.metrics = metrics

        for metric in available_metrics:
            self.params[metric] = []

        self.num_epochs = epochs
        self.batch_size = batch_size
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(train_dataloader):
                
                train = torch.autograd.Variable(images.view(len(images),6,16,16,16))
                labels = torch.autograd.Variable(labels)
                # Clear gradients
                self.optimizer.zero_grad()
                # Forward propagation
                outputs = self(train.float().to(self.device)).to(self.device)
                # Calculate loss
                loss = self.error(outputs, labels.float().unsqueeze(1).to(self.device)).to(self.device)
                # Calculating gradients
                loss.backward()
                # Update parameters
                self.optimizer.step()
                
                self.count += 1

                print(f"Epoch {epoch+1}/{self.num_epochs}: Batch {i}/{len_training//self.batch_size} \tLoss: {loss.data}")
            
            try:
                self.scheduler.step()
            except:
                pass

            # Calculate metrics         
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
                        TP += 1

                    elif outputs[idx] < 0.5 and labels[idx] == 1:
                        FN += 1

                    elif outputs[idx] < 0.5 and labels[idx] == 0:
                        TN += 1
                    
                    elif outputs[idx] > 0.5 and labels[idx] == 0:
                        FP += 1

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

            self.params['accuracy'].append(accuracy)
            self.params['loss'].append(loss)
            self.params['TP'].append(TP)
            self.params['FP'].append(FP)
            self.params['TN'].append(TN)
            self.params['FN'].append(FN)
            self.params['precision'].append(precision)
            self.params['negative_predictive_value'].append(negative_predictive_value)
            self.params['sensitivity'].append(sensitivity)
            self.params['recall'].append(sensitivity)
            self.params['f1'].append(f1)
            self.params['f2'].append(f2)
            self.print_params()
            if epoch % 10 and epoch != 0:
                self.save_model()
            elif self.params['accuracy'][-1] > 0.7:
                self.save_model()
            #self.check_callbacks()


        self.save_model(True)

class MCDCNN(DCNN):

    def __init__(self, name, path, n_channels):

        super(DCNN, self).__init__()

        self.count = 0
        self.channels = nn.ModuleList()
        self.n_channels = n_channels
        self.layers = nn.ModuleList()
        self.name = name
        self.save = path
        self.params = {}
        self.callbacks = []

        self.float()

        for _ in range(self.n_channels):
            self.channels.extend([Channel()])
    
    def add_layers_to_channels(self, channels, layers):
        for layer in layers:
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
