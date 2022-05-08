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

from tfm.big_dcnn import DROPOUT_CLASSIFIER, DROPOUT_FEATURES

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
    input_parameters = ez.utils.parse_inputs(os.path.join("/", "HDD2", "tfm", METAL, "Inputs", file))

    # Global parameters that should not be changed
    TRAINING_DATA = input_parameters['training_data']
    BATCH_SIZE = int(input_parameters['batch_size'])
    NUM_EPOCHS = int(input_parameters['num_epochs'])
    LEARNING_RATE = float(input_parameters['lr'])
    WEIGHT_DECAY = float(input_parameters['weight_decay'])
    MODEL_NAME = input_parameters['name']
    PATH = input_parameters['experiment']
    NUM_CLASSES = int(input_parameters['num_classes'])
    NUM_CHANNELS = int(input_parameters['num_channels'])
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

    # Obtain training and validation data
    training_folds = []
    validate_folds = []
    for file in os.listdir(os.path.join(".", "Datasets", TRAINING_DATA+"s")):
        if file.endswith(".csv") and file.startswith(TRAINING_DATA + "_train"):
            training_folds.append(file)
        elif file.endswith(".csv") and file.startswith(TRAINING_DATA + "_val"):
            validate_folds.append(file)
    training_folds.sort()
    validate_folds.sort()
    file1 = training_folds[0]
    file2 = validate_folds[0]

    # Define the model
    dcnn_model = ez.architectures.dcnn()
    dcnn_model.features = nn.Sequential(
        ez.layers.conv3d(in_channels=6, out_channels=64, kernel_size=3, stride=1, dropout=DROPOUT_FEATURES,
                            batch_norm=BATCH_NORM, padding='same', activation_function=ACTIVATION_FUNCTION),
        ez.layers.fire3d(in_channels=64, squeeze_channels=32, expand1x1_channels=64, expandnxn_channels=64, 
                            dropout=DROPOUT_FEATURES, batch_norm=BATCH_NORM, activation_function=ACTIVATION_FUNCTION,
                            expand_kernel=7),
        ez.layers.conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, dropout=DROPOUT_FEATURES, 
                            batch_norm=BATCH_NORM, padding='same', activation_function=ACTIVATION_FUNCTION),
        nn.MaxPool3d(kernel_size=2),
        ez.layers.fire3d(in_channels=128, squeeze_channels=64, expand_kernel=5, expand1x1_channels=192, 
                            expandnxn_channels=64, dropout=DROPOUT_FEATURES, batch_norm=BATCH_NORM, 
                            activation_function=ACTIVATION_FUNCTION),
        ez.layers.conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, dropout=DROPOUT_FEATURES, 
                            batch_norm=BATCH_NORM, padding='same', activation_function=ACTIVATION_FUNCTION),
        nn.MaxPool3d(kernel_size=2),
        ez.layers.fire3d(in_channels=256, squeeze_channels=128, expand_kernel=3, expand1x1_channels=384, 
                            expandnxn_channels=128, dropout=DROPOUT_FEATURES, batch_norm=BATCH_NORM, 
                            activation_function=ACTIVATION_FUNCTION),
        nn.MaxPool3d(kernel_size=2),

    )
    dcnn_model.flatten = nn.Flatten()
    dcnn_model.classifier = nn.Sequential(
        ez.layers.dense(in_features=4096, out_features=1024, dropout=DROPOUT_CLASSIFIER,
                            activation_function=ACTIVATION_FUNCTION),
        ez.layers.dense(in_features=1024, out_features=512, dropout=DROPOUT_CLASSIFIER,
                            activation_function=ACTIVATION_FUNCTION),
        ez.layers.dense(in_features=512, out_features=256, dropout=DROPOUT_CLASSIFIER,
                            activation_function=ACTIVATION_FUNCTION),
        ez.layers.dense(in_features=256, out_features=128, dropout=DROPOUT_CLASSIFIER,
                            activation_function=ACTIVATION_FUNCTION),
        ez.layers.dense(in_features=128, out_features=1, dropout=0,
                            activation_function=nn.Sigmoid()),
        
    )
    print(dcnn_model)

    trainer = ez.architectures.trainer(dcnn_model, MODEL_NAME, os.path.join(".",METAL, PATH))
    trainer.callbacks.append(ez.callbacks.checkpoint(metric='accuracy', target=0.70, model=trainer))
    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    trainer.compile(optimizer, nn.BCELoss(), device=DEVICE)
    trainer.input_shape = (6,16,16,16)

    training_data = CustomDataset(os.path.join(".", "Datasets", TRAINING_DATA+"s", file1), 
                                    os.path.join(".",METAL, PATH), 
                                    MODEL_NAME, DATASET_PROPORTION, dcnn_model, trainer, DEVICE)
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = CustomDataset(os.path.join(".", "Datasets", TRAINING_DATA+"s", file2), 
                                    os.path.join(".",METAL, PATH), 
                                    MODEL_NAME, DATASET_PROPORTION, dcnn_model, trainer, DEVICE)
    validate_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)

    with open(os.path.join(".", METAL, PATH, MODEL_NAME, "input_params"), "w") as fo:
        for key, value in input_parameters.items():
            fo.write(f"{key}: {value}\n")

    print(f"Number of trainable parameters: {trainer.count_parameters()}")
    start = time.time()
    trainer.train_model(train_dataloader, validate_dataloader,len(training_data), NUM_EPOCHS, BATCH_SIZE, METRICS)
    end = time.time()

    with open(os.path.join(".",METAL, PATH, MODEL_NAME, "Time"), "w") as fo:
        fo.write(f"Time to train {(end-start)/60} min")
    
        

