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

def xflip(img: torch.Tensor) -> torch.Tensor:
    return img.flip(-1)

def yflip(img: torch.Tensor) -> torch.Tensor:
    return img.flip(-2)

def zflip(img: torch.Tensor) -> torch.Tensor:
    return img.flip(-3)

class random_x_flip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return xflip(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class random_y_flip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return xflip(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class random_z_flip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return xflip(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"