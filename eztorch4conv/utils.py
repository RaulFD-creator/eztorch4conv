import json
import pandas as pd
import random
import torch
from torch.utils.data import Dataset

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
    
    with open(path_to_arguments, 'r') as j:
        return json.loads(j.read())
