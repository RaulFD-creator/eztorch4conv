import torch
import torch.nn as nn

class conv3d(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int=3 or tuple, stride : int=1,  
                dropout : float=0, batch_norm : bool=False, padding : str='valid',
                activation_function : torch.Tensor=nn.ELU(inplace=True)) -> None:
        super().__init__()

        # Parsing inputs
        if not isinstance(padding, int): self.padding = kernel_size // 2 if padding == 'same' else 0
        else: self.padding = padding

        self.batch_norm = nn.BatchNorm3d(in_channels) if batch_norm else None
        self.main_layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride, self.padding)
        self.activation_function = activation_function
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if self.batch_norm is not None:
            return self.dropout(self.activation_function((self.main_layer(self.batch_norm(x)))))
        else:
            return self.dropout(self.activation_function((self.main_layer(x))))

class dense(nn.Module):
    def __init__(self, in_features : int, out_features : int, dropout : float=0, 
                activation_function : torch.Tensor=nn.ELU()) -> None:
        super().__init__()
        self.main_layer = nn.Linear(in_features, out_features)
        self.activation_function = activation_function
        self.dropout = nn.Dropout(dropout)

        self.output_shape = out_features


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return(self.dropout(self.activation_function(self.main_layer(x))))

class fire3d(nn.Module):
    def __init__(self, in_channels: int, squeeze_channels: int, expand1x1_channels: int, expandnxn_channels: int,
                batch_norm : bool=True, dropout : float=0, activation_function = nn.ELU(inplace=True),
                expand_kernel : int=3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.squeeze = nn.Conv3d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = activation_function
        self.expand1x1 = nn.Conv3d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = activation_function
        self.expandnxn = nn.Conv3d(squeeze_channels, expandnxn_channels, kernel_size=expand_kernel, padding=expand_kernel//2)
        self.expandnxn_activation = activation_function
        if batch_norm: self.batch_norm = nn.BatchNorm3d(in_channels)
        if batch_norm: self.batch_norm_flag = batch_norm
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(self.batch_norm(x))) if self.batch_norm_flag else self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expandnxn_activation(self.expandnxn(x))], 1
        )