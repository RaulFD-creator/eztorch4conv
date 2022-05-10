from typing import List, Any
import torch
import torch.nn as nn

class conv3d(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int=3 or tuple, stride : int=1,  
                dropout : float=0, batch_norm : bool=False, padding : str='valid',
                activation_function : torch.Tensor=nn.ELU(inplace=True)) -> None:
        super().__init__()

        # Parsing inputs
        if not (isinstance(padding, int) or isinstance(padding, tuple)): self.padding = kernel_size // 2 if padding == 'same' else 0
        else: self.padding = padding

        self.batch_norm = nn.BatchNorm3d(in_channels) if batch_norm else None
        self.main_layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride, self.padding)
        self.activation_function = activation_function
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(x) if self.batch_norm is not None else x
        return self.dropout(self.activation_function((self.main_layer(x)))) 

class dense(nn.Module):
    def __init__(self, in_features : int, out_features : int, dropout : float=0, 
                activation_function : torch.Tensor=nn.ELU(), batch_norm : bool=False) -> None:
        super().__init__()
        self.main_layer = nn.Linear(in_features, out_features)
        self.activation_function = activation_function
        self.batch_norm = nn.BatchNorm1d(in_features) if batch_norm else None
        self.dropout = nn.Dropout(dropout)

        self.output_shape = out_features


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(x) if self.batch_norm is not None else x
        return self.dropout(self.activation_function((self.main_layer(x)))) 

class fire3d(nn.Module):
    def __init__(self, in_channels : int, squeeze_channels : int, expand_1x1x1_channels : int, expand_nxnxn_channels : int,
                batch_norm : bool=True, dropout : float=0, activation_function = nn.ELU(inplace=True),
                expand_kernel : int=3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.squeeze = nn.Conv3d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = activation_function
        self.expand1x1 = nn.Conv3d(squeeze_channels, expand_1x1x1_channels, kernel_size=1)
        self.expand1x1_activation = activation_function
        self.expandnxn = nn.Conv3d(squeeze_channels, expand_nxnxn_channels, kernel_size=expand_kernel, padding=expand_kernel//2)
        self.expandnxn_activation = activation_function
        if batch_norm: self.batch_norm = nn.BatchNorm3d(in_channels)
        if batch_norm: self.batch_norm_flag = batch_norm
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(self.batch_norm(x))) if self.batch_norm_flag else self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expandnxn_activation(self.expandnxn(x))], 1
        )

class InceptionD(nn.Module):
    def __init__(self, in_channels: int, neurons_nxnxn : int=192, neurons_3x3x3 : int=320, kernel_size : int=7) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = conv3d
        self.branch3x3x3_1 = conv_block(in_channels, neurons_nxnxn, kernel_size=1, batch_norm=True)
        self.branch3x3x3_2 = conv_block(neurons_nxnxn, neurons_3x3x3, kernel_size=3, stride=2, batch_norm=True)

        pad = kernel_size // 2
        self.branch7x7x7_1 = conv_block(in_channels, neurons_nxnxn, kernel_size=1, batch_norm=True)
        self.branch7x7x7_2 = conv_block(neurons_nxnxn, neurons_nxnxn, kernel_size=(1, 1, kernel_size), padding=(0, 0, pad), batch_norm=True)
        self.branch7x7x7_3 = conv_block(neurons_nxnxn, neurons_nxnxn, kernel_size=(kernel_size, 1, 1), padding=(pad, 0, 0), batch_norm=True)
        self.branch7x7x7_4 = conv_block(neurons_nxnxn, neurons_nxnxn, kernel_size=(1, kernel_size, 1), padding=(0, pad, 0), batch_norm=True)
        self.branch7x7x7_5 = conv_block(neurons_nxnxn, neurons_nxnxn, kernel_size=3, stride=2, batch_norm=True, padding='valid')

    def _forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        branch3x3 = self.branch3x3x3_1(x)
        branch3x3 = self.branch3x3x3_2(branch3x3)

        branch7x7x7 = self.branch7x7x7_1(x)
        branch7x7x7 = self.branch7x7x7_2(branch7x7x7)
        branch7x7x7 = self.branch7x7x7_3(branch7x7x7)
        branch7x7x7 = self.branch7x7x7_4(branch7x7x7)
        branch7x7x7 = self.branch7x7x7_5(branch7x7x7)

        branch_pool = nn.F.max_pool3d(x, kernel_size=2, stride=2)
        outputs = [branch3x3, branch7x7x7, branch_pool]
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

