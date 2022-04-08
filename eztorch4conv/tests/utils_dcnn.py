import torch
import pandas as pd
import os
import random
from torch.utils.data import Dataset


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
        os.system(f"touch Models/{model_name}/data_division.log")
        with open(f"Models/{model_name}/data_division.log","a") as fo:
            fo.write(f"{output}\n")
        with open(f"Models/{model_name}/input.data","w") as fo:
            for key, value in input_parameters.items():
                fo.write(f"{key}: {value}\n")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Suffle df to mitigate overfitting
        self.labels = self.labels.sample(frac=1, random_state=2812).reset_index(drop=True)  
        image = torch.load(self.labels.iloc[idx, 0]).float()
        label = torch.tensor(self.labels.iloc[idx, 1]).float()
        return image, label

