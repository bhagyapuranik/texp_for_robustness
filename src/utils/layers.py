from typing import Union, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from math import floor
import numpy as np
from pathlib import Path


class ImplicitNormalizationConv(nn.Conv2d):
    # this version does not take absolute value
    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        inp_norm = False
        if inp_norm :
            divisor = torch.norm(x.reshape(x.shape[0], -1), dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)  
            x = x*np.sqrt(x.numel()/x.shape[0])/(divisor+1e-6)
        
        weight_norms = (self.weight**2).sum(dim=(1, 2, 3), keepdim=True).transpose(0, 1).sqrt()

        conv = super().forward(x)
        return conv/(weight_norms+1e-6)


    
class Normalize(nn.Module):
    """Data normalizing class as torch.nn.Module

    Attributes:
        mean (float): Mean value of the training dataset.
        std (float): Standard deviation value of the training dataset.

    """

    def __init__(self, mean, std):
        """
        Args:
            mean (float): Mean value of the training dataset.
            std (float): Standard deviation value of the training dataset.
        """
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        """
        Args:
            x (tensor batch): Input tensor.
        Returns:
            Normalized data.
        """
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class TexpNormalization(nn.Module):
    r"""Applies tilted exponential normalization over an input signal composed of several input
    planes. (Tilted Softmax Block)

    Args:
        tilt: Tilt of the exponential function, must be > 0.
    """

    def __init__(
        self,  
        tilt: float = 1.0,
        texp_across_filts_only: bool = False
        ) -> None:
        super(TexpNormalization, self).__init__()

        self.tilt = tilt
        self.texp_across_filts_only = texp_across_filts_only


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Returns softmax of input tensor, for each image in batch"""
        if self.texp_across_filts_only:
            return torch.exp(self.tilt*input)/torch.sum(torch.exp(self.tilt*input),dim=(1),keepdim=True)
        else:
            return torch.exp(self.tilt*input)/torch.sum(torch.exp(self.tilt*input),dim=(1,2,3),keepdim=True)

    def __repr__(self) -> str:
        s = "TexpNormalization("
        s += f'tilt={self.tilt}_filts_only_{self.texp_across_filts_only}'
        s += ")"
        return s



class AdaptiveThreshold(nn.Module):
    r"""
    Thresholds values x[x>threshold]
    """

    def __init__(self, std_scalar: float = 0.5, mean_plus_std: bool=True) -> None:
        super(AdaptiveThreshold, self).__init__()
        self.std_scalar = std_scalar
        self.means_plus_std = mean_plus_std

    def _thresholding(self, x, threshold):
        return x*(x > threshold)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        means = input.mean(dim=(2, 3), keepdim=True)
        stds = input.std(dim=(2, 3), keepdim=True)
        if self.means_plus_std:
            return self._thresholding(input, means + self.std_scalar*stds)
        else:
            return self._thresholding(input, means)

    def __repr__(self) -> str:
        if self.means_plus_std:
            s = f"AdaptiveThreshold(mean_plus_std, std_scale={self.std_scalar})"
        else:
            s = f"AdaptiveThreshold(mean_scalar={self.mean_scalar})"
        return s
    
