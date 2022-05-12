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
import torchvision
import torch.nn as nn
import torch
import os

class CustomDataset(Dataset):

    def __init__(self, annotations_file, proportion, device='cpu'):
        self.labels = pd.read_csv(annotations_file)
        self.labels = self.labels.sample(frac=1, random_state=2812).reset_index(drop=True)
        self.device = device
        random.seed(2812)
        to_remove = []
        for idx, _ in self.labels.iterrows():
            if random.random() < 1-proportion: to_remove.append(idx)
        self.labels.drop(to_remove, axis=0, inplace=True)

        
        images = list(self.labels.iloc[:, 0])
        ground_truths = list(self.labels.iloc[:, 1])
        self.labels = [(images[i],  ground_truths[i]) for i in range(len(images))]

        # Augmentation
        torch.manual_seed(2812)
        self.x_flip = ez.utils.random_x_flip(p=0.5)
        self.y_flip = ez.utils.random_y_flip(p=0.5)
        self.z_flip = ez.utils.random_z_flip(p=0.5)
        self.rotation = torchvision.transforms.RandomRotation(degrees=10)


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):

        item = self.labels[idx]
        image = (self.z_flip(self.y_flip(self.x_flip(torch.load(item[0]).float().to(self.device)))))
        label = item[1]

        return image, label

def test_eztorch4conv_imported():
    input_parameters = ez.utils.parse_inputs("eztorch4conv/data/input_params.json")

    # Global parameters that should not be changed
    TRAINING_DATA = input_parameters['training_data']
    BATCH_SIZE = int(input_parameters['batch_size'])
    NUM_EPOCHS = int(input_parameters['num_epochs'])
    LEARNING_RATE = float(input_parameters['lr'])
    WEIGHT_DECAY = float(input_parameters['weight_decay'])
    MODEL_NAME = input_parameters['name']
    DEVICE = input_parameters['device']
    DATASET_PROPORTION = float(input_parameters['proportion'])
    METRICS = input_parameters['metrics']
    DROPOUT_FEATURES = float(input_parameters['d_features'])
    DROPOUT_CLASSIFIER = float(input_parameters['d_classifier'])
    BATCH_NORM = bool(input_parameters['batch_norm'])
    ACTIVATION_FUNCTION = input_parameters['activation']
    if ACTIVATION_FUNCTION == "ReLU":
        ACTIVATION_FUNCTION = nn.ReLU(inplace=True)
    elif ACTIVATION_FUNCTION == "LeakyReLU":
        ACTIVATION_FUNCTION = nn.LeakyReLU(inplace=True)
    else:
        ACTIVATION_FUNCTION = nn.ELU(inplace=True)

    # Define the model
    dcnn_model = ez.architectures.dcnn()
    dcnn_model.features = nn.Sequential(
        ez.layers.conv3d(in_channels=6, out_channels=12, kernel_size=3, stride=1, dropout=DROPOUT_FEATURES,
                            batch_norm=BATCH_NORM, padding='same', activation_function=ACTIVATION_FUNCTION),
        ez.layers.fire3d(in_channels=12, squeeze_channels=32, expand1x1x1_channels=24, expandnxnxn_channels=24, 
                            dropout=DROPOUT_FEATURES, batch_norm=BATCH_NORM, activation_function=ACTIVATION_FUNCTION,
                            expand_kernel=5),
        ez.layers.InceptionD(in_channels=48, neurons_nxnxn=64, neurons_3x3x3=64, kernel_size=5),
        nn.MaxPool3d(kernel_size=2)
    )

    dcnn_model.flatten = nn.Flatten()
    dcnn_model.classifier = nn.Sequential(
        ez.layers.dense(in_features=128, out_features=1, dropout=0,
                            activation_function=nn.Sigmoid(), batch_norm=True),
        
    )

    trainer = ez.architectures.trainer(dcnn_model, MODEL_NAME, os.path.join("."), save_files=False)
    trainer.callbacks.append(ez.callbacks.checkpoint(metric='accuracy', target=0.70, model=trainer))
    trainer.callbacks.append(ez.callbacks.early_stop(metric='accuracy', target=1e-6))
    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    trainer.compile(optimizer, nn.BCELoss(), device=DEVICE)
    trainer.input_shape = (6,16,16,16)

    training_data = CustomDataset(os.path.join("..", "data", TRAINING_DATA), DATASET_PROPORTION, DEVICE)
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = CustomDataset("..", "data", TRAINING_DATA, DATASET_PROPORTION, DEVICE)
    validate_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)

    trainer.train_model(train_dataloader, validate_dataloader,len(training_data), NUM_EPOCHS, BATCH_SIZE, METRICS)    

