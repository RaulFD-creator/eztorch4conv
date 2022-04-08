"""
Unit and regression test for the eztorch4conv package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

#import eztorch4conv
import torch.nn as nn
import numpy as np
import torch
import eztorch4conv as ez
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from utils_dcnn import CustomDataset
import os

def test_eztorch4conv_imported():

    with open("./dataset/reccord.csv", "w") as of:
        of.write("Path,Binding\n")

    for i in range(100):
        print(i)
        array = np.random.rand(6,16,16,16)
        torch_array = torch.from_numpy(array)
        with open("./reccord.csv", "a") as of:
            of.write(f"./{i}.pt,{1 if i > 40 else 0}\n")
    torch.save(torch_array, f'{i}.pt')

    model_num = 0
    for LEARNING_RATE in [5e-6]:
        for WEIGHT_DECAY in [0]:
            for BATCH_SIZE in [10]:
                for ARCHITECTURE in ['mcdcnn']:
                    input_parameters = {
                        'learning_rate' : LEARNING_RATE,
                        'num_epochs' : 2,
                        'batch_size' : BATCH_SIZE,
                        'weight_decay' : WEIGHT_DECAY,
                        'architecture' : ARCHITECTURE,
                        'model_name' : ARCHITECTURE + '_' + str(model_num//2),
                        'num_classes' : 1,
                        'num_channels' : 6 if ARCHITECTURE == 'dcnn' else 1,
                        'output_flatten' : 512 if ARCHITECTURE == 'dcnn' else 3072, 
                        'device' : 'cpu',
                        'dataset_proportion' : 0.25,
                        'metrics' : 'all',
                        'training_data' : "./dataset/reccord.csv",
                        "validation_data" : "./dataset/reccord.csv"

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
                    OUTPUT_FLATTEN = input_parameters['output_flatten']

                    if ARCHITECTURE == "dcnn":
                        print("Using DCNN architecture")
                        # Create DCNN
                        model = ez.architectures.DCNN(MODEL_NAME, os.path.join(".", "Models"))
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
                                            ez.layers.flatten(OUTPUT_FLATTEN),
                                            ez.layers.dense(neurons=512, dropout=0.5),
                                            ez.layers.dense(neurons=512, dropout=0.2),
                                            ez.layers.dense(neurons=256, dropout=0.4),
                                            ez.layers.dense(neurons=128, dropout=0.5),
                                            ez.layers.dense(neurons=64, dropout=0.5),
                                            ez.layers.dense(neurons=32, dropout=0.5),
                                            ez.layers.dense(neurons=16, dropout=0.5),
                                            ez.layers.dense(neurons=1, dropout=0, activation_function=nn.Sigmoid())
                        ])


                        

                    elif ARCHITECTURE == "mcdcnn" or ARCHITECTURE == "mc-dcnn":
                        print("Using MC-DCNN architecture")
                        model = ez.architectures.MCDCNN(MODEL_NAME, os.path.join(".", "Models"), 6, (6,16,16,16))
                        model.add_layers_to_channels('all', 
                            [
                                            ez.layers.conv3d(input_shape=(NUM_CHANNELS,16,16,16), neurons=32, 
                                                            conv_kernel=5, conv_padding=2, pooling_type=None, 
                                                            dropout=0),
                                            ez.layers.conv3d(neurons=64, conv_kernel=1,
                                                            pooling_type='max', dropout=0.25, batch_norm=nn.BatchNorm3d(64)),
                                            ez.layers.conv3d(neurons=128, conv_kernel=3, padding='same',
                                                            pooling_type=None, dropout=0.25, batch_norm=nn.BatchNorm3d(128)),
                                            ez.layers.conv3d(neurons=128, conv_kernel=3,
                                                            padding='same', pooling_type='max', dropout=0.2),
                                            ez.layers.conv3d(neurons=128, conv_kernel=3,
                                                            conv_padding=1, pooling_type=None, dropout=0.2),
                                            ez.layers.conv3d(neurons=256, conv_kernel=3, batch_norm=nn.BatchNorm3d(256),
                                                            conv_padding=1, pooling_type=None, dropout=0.2),
                                            ez.layers.conv3d(neurons=256, conv_kernel=3, 
                                                            conv_padding=1, pooling_type='max', dropout=0.25),
                                            ez.layers.conv3d(neurons=512, conv_kernel=3, batch_norm=nn.BatchNorm3d(512),
                                                            conv_padding=1, pooling_type='max', dropout=0.25),
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
                                            ez.layers.dense(neurons=16, dropout=0.5),
                                            ez.layers.dense(neurons=1, dropout=0, activation_function=nn.Sigmoid())
                            ])


                    model.define_loss(nn.BCELoss())
                    # Optimization with ADAM and L2 regularization
                    if WEIGHT_DECAY == 1:
                        model.define_optimizer(torch.optim.Adam(model.parameters(), lr=LEARNING_RATE))
                    else:
                        model.define_optimizer(torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY))

                    training_data = CustomDataset(TRAINING_DATA, MODEL_NAME, DATASET_PROPORTION, input_parameters)
                    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
                    validation_data = CustomDataset(VALIDATION_DATA, MODEL_NAME, DATASET_PROPORTION, input_parameters)
                    validate_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)

                    #print(model)
                    print(f"Number of trainable parameters: {model.count_parameters()}")

                    model.device = DEVICE
                    model.train_model(train_dataloader, validate_dataloader,len(training_data), 
                                NUM_EPOCHS, BATCH_SIZE, METRICS)

    
        

