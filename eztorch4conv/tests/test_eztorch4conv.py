"""
Unit and regression test for the eztorch4conv package.
"""

# Import package, test suite, and other packages as needed
import sys
import pandas as pd
import pytest
import random
import torch.nn as nn
import numpy as np
import torch
import eztorch4conv as ez
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import os

class CustomDataset(Dataset):
    def __init__(self, annotations_file, model_name, proportion, input_parameters):
        random.seed(2812)
        self.labels = pd.read_csv(annotations_file)
        self.labels = self.labels.sample(frac=1, random_state=2812).reset_index(drop=True)

        for idx, _ in self.labels.iterrows():
        	if random.random() < 1-proportion:
        		self.labels.drop(idx, axis=0, inplace=True)
        		
        majority = self.labels["Binding"].value_counts()[0]
        minority = self.labels["Binding"].value_counts()[1]

        difference = majority - minority
        counter = 0
        for idx, _ in self.labels[self.labels["Binding"] == 0].iterrows():
            if counter < difference:
                self.labels.drop(idx, axis=0, inplace=True)
                counter += 1

        output = self.labels["Binding"].value_counts()
        #os.system(f"touch Models/{model_name}/data_division.log")
        #with open(f"Models/{model_name}/data_division.log","a") as fo:
        #    fo.write(f"{output}\n")
        #with open(f"Models/{model_name}/input.data","w") as fo:
        #    for key, value in input_parameters.items():
        #        fo.write(f"{key}: {value}\n")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Suffle df to mitigate overfitting
        self.labels = self.labels.sample(frac=1, random_state=2812).reset_index(drop=True)  
        image = torch.load(self.labels.iloc[idx, 0]).float()
        label = torch.tensor(self.labels.iloc[idx, 1]).float()
        return image, label


def test_eztorch4conv_imported():


    model_num = 0
    for LEARNING_RATE in [5e-6]:
        for WEIGHT_DECAY in [0]:
            for BATCH_SIZE in [10]:
                for ARCHITECTURE in ['dcnn', 'mcdcnn']:
                    input_parameters = {
                        'learning_rate' : LEARNING_RATE,
                        'num_epochs' : 2,
                        'batch_size' : BATCH_SIZE,
                        'weight_decay' : WEIGHT_DECAY,
                        'architecture' : ARCHITECTURE,
                        'model_name' : ARCHITECTURE + '_' + str(model_num//2),
                        'num_classes' : 1,
                        'num_channels' : 6 if ARCHITECTURE == 'dcnn' else 1,
                        'device' : 'cpu',
                        'dataset_proportion' : 0.25,
                        'metrics' : 'all',
                        'training_data' : "eztorch4conv/tests/dataset/reccord.csv",
                        "validation_data" : "eztorch4conv/tests/dataset/reccord.csv"

                    }
                    model_num += 1

                    BATCH_SIZE = int(input_parameters["batch_size"])
                    NUM_EPOCHS = int(input_parameters["num_epochs"])
                    LEARNING_RATE = float(input_parameters["learning_rate"])
                    WEIGHT_DECAY = float(input_parameters["weight_decay"])
                    ARCHITECTURE = input_parameters["architecture"]
                    MODEL_NAME = input_parameters["model_name"]
                    NUM_CLASSES = int(input_parameters["num_classes"])
                    NUM_CHANNELS = int(input_parameters["num_channels"])
                    DEVICE = input_parameters["device"]
                    DATASET_PROPORTION = input_parameters['dataset_proportion']
                    METRICS = input_parameters["metrics"]
                    TRAINING_DATA = input_parameters["training_data"]
                    VALIDATION_DATA = input_parameters["validation_data"]

                    if ARCHITECTURE == "dcnn":
                        print("Using DCNN architecture")
                        # Create DCNN
                        model = ez.architectures.DNN(MODEL_NAME, os.path.join(".", "Models"), False)
                        model.add_layers([
                                            ez.layers.conv3d(input_shape=(NUM_CHANNELS,16,16,16), neurons=32, 
                                                            conv_kernel=5, padding='same', pooling_type='max', 
                                                            dropout=0, batch_norm=True),
                                            ez.layers.conv3d(neurons=64, conv_kernel=1, padding='same',
                                                            pooling_type='none', dropout=0.25),
                                            ez.layers.conv3d(neurons=128, conv_kernel=3, padding='same',
                                                            pooling_type=None, dropout=0.25),
                                            ez.layers.conv3d(neurons=128, conv_kernel=3,
                                                            padding='same', pooling_type='none', dropout=0.2),
                                            ez.layers.conv3d(neurons=128, conv_kernel=3,
                                                            padding='same', pooling_type=None, dropout=0.2),
                                            ez.layers.conv3d(neurons=256, conv_kernel=3, batch_norm=False,
                                                            padding='same', pooling_type=None, dropout=0.2),
                                            ez.layers.conv3d(neurons=256, conv_kernel=3, 
                                                            padding='same', pooling_type='max', dropout=0.25),
                                            ez.layers.conv3d(neurons=512, conv_kernel=3, batch_norm=False,
                                                            padding='same', pooling_type='max', dropout=0.25),
                                            ez.layers.flatten(),
                                            ez.layers.dense(neurons=512, dropout=0.5),
                                            ez.layers.dense(neurons=512, dropout=0.2),
                                            ez.layers.dense(neurons=256, dropout=0.4),
                                            ez.layers.dense(neurons=128, dropout=0.5),
                                            ez.layers.dense(neurons=64, dropout=0.5),
                                            ez.layers.dense(neurons=32, dropout=0.5),
                                            ez.layers.dense(neurons=16, dropout=0.5),
                        ])
                        model.add_layers(ez.layers.dense(neurons=1, dropout=0, activation_function=nn.Sigmoid())
)
                        

                    elif ARCHITECTURE == "mcdcnn" or ARCHITECTURE == "mc-dcnn":
                        print("Using MC-DCNN architecture")
                        model = ez.architectures.MCDNN(MODEL_NAME, os.path.join(".", "Models"), n_channels=6, input_shape=(6,16,16,16), 
                                                        save_files=False)
                        model.add_layers_to_channels('all', 
                            [
                                            ez.layers.conv3d(input_shape=(NUM_CHANNELS,16,16,16), neurons=32, 
                                                            conv_kernel=5, padding='same', pooling_type='max', 
                                                            dropout=0, batch_norm=True),
                                            ez.layers.conv3d(neurons=64, conv_kernel=1, padding='same',
                                                            pooling_type='none', dropout=0.25),
                                            ez.layers.conv3d(neurons=128, conv_kernel=3, padding='same',
                                                            pooling_type=None, dropout=0.25),
                                            ez.layers.conv3d(neurons=128, conv_kernel=3,
                                                            padding='same', pooling_type='none', dropout=0.2),
                                            ez.layers.conv3d(neurons=128, conv_kernel=3,
                                                            padding='same', pooling_type=None, dropout=0.2),
                                            ez.layers.conv3d(neurons=256, conv_kernel=3, batch_norm=False,
                                                            padding='same', pooling_type=None, dropout=0.2),
                                            ez.layers.conv3d(neurons=256, conv_kernel=3, 
                                                            padding='same', pooling_type='max', dropout=0.25),
                                            ez.layers.conv3d(neurons=512, conv_kernel=3, batch_norm=False,
                                                            padding='same', pooling_type='max', dropout=0.25),
                                            ez.layers.flatten(),
                        ])
                        model.add_layers(
                            [
                                            ez.layers.dense(neurons=512, dropout=0.5),
                                            ez.layers.dense(neurons=512, dropout=0.2),
                                            ez.layers.dense(neurons=256, dropout=0.4),
                                            ez.layers.dense(neurons=128, dropout=0.5),
                                            ez.layers.dense(neurons=64, dropout=0.5),
                                            ez.layers.dense(neurons=32, dropout=0.5),
                                            ez.layers.dense(neurons=16, dropout=0.5)
                            ])
                        model.add_layers([ez.layers.dense(neurons=1, dropout=0, activation_function=nn.Sigmoid())])


                    model.add_callbacks([ez.callbacks.early_stop(metric='accuracy', target=0.7, model=model), 
                                        ez.callbacks.checkpoint(metric='accuracy', target=0.5, model=model)])

                    training_data = CustomDataset(TRAINING_DATA, MODEL_NAME, DATASET_PROPORTION, input_parameters)
                    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
                    validation_data = CustomDataset(VALIDATION_DATA, MODEL_NAME, DATASET_PROPORTION, input_parameters)
                    validate_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)

                    model.compile(torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY), nn.BCELoss())

                    #print(model)
                    print(f"Number of trainable parameters: {model.count_parameters()}")

                    model.device = DEVICE
                    checkpoints, history = model.train_model(train_dataloader, validate_dataloader,len(training_data), 
                                NUM_EPOCHS, BATCH_SIZE, METRICS)

    
        

