from typing import Any, Callable, List, Optional, Type, Union, Iterable

from torch import Tensor

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from math import floor

from .layers import AdaptiveThreshold, Normalize, TexpNormalization, ImplicitNormalizationConv

vgg_depth_dict = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class TEXP_VGG(nn.Module):
    def __init__(
            self, vgg_depth: str = "VGG16", conv_layer_name: str = "conv2d", conv_layer_type: type = nn.Conv2d, threshold: Optional[float] = None, 
            texp_count: int = 0, dataset_name: str = 'cifar10', 
            norm_layer_type = 'texp', norm_layer_tilt = 0.192, adap_thresh=True, 
            texp_across_filts_only = True, threshold_mean_plus_std = True) -> None:
        
        super().__init__()
        if dataset_name == 'CIFAR10':
            self.normalize = Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2471, 0.2435, 0.2616))
        else:
            raise NotImplementedError

        self.vgg_depth = vgg_depth
        self.conv_layer_name = conv_layer_name
        self.conv_layer_type = conv_layer_type
        self.threshold = threshold
        self.texp_count = texp_count
        self.norm_layer_type = norm_layer_type
        self.norm_layer_tilt = norm_layer_tilt
        self.adap_thresh = adap_thresh
        self.texp_across_filts_only = texp_across_filts_only
        self.threshold_mean_plus_std = threshold_mean_plus_std
        

        self.features = self.make_layers(vgg_depth_dict[vgg_depth])
        if dataset_name == 'CIFAR10':
            self.classifier = nn.Linear(512, 10)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, layer_list: Iterable[Union[str, int]]) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = 3
        for layer_i, layer_info in enumerate(layer_list):
            if isinstance(layer_info, str):
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            else:
                layer_width = layer_info
                threshold_layer = []
                
                
                if layer_i <= self.texp_count:
                    if self.norm_layer_type == 'texp':
                        normalization_layer: nn.Module = TexpNormalization(tilt=self.norm_layer_tilt[layer_i], texp_across_filts_only = self.texp_across_filts_only)
                else:
                    normalization_layer = nn.BatchNorm2d(layer_width)
                
                if self.adap_thresh:
                    if (self.threshold is not None and self.threshold >= 0.0) and layer_i <= self.texp_count:
                        threshold_layer = [AdaptiveThreshold(
                            std_scalar=self.threshold, mean_plus_std=self.threshold_mean_plus_std)]

                
                conv_layer = self.conv_layer_type(
                    in_channels, layer_width, kernel_size=3, padding=1, bias=False)
                
                if self.norm_layer_type == 'texp' and layer_i <= self.texp_count:
                    layers += [conv_layer, normalization_layer]+threshold_layer
                else:
                    layers += [conv_layer,  nn.ReLU(inplace=False),
                           normalization_layer]+threshold_layer

                in_channels = layer_width
        return nn.Sequential(*layers)
               


    @property
    def name(self) -> str:
        if self.conv_layer_name != "conv2d":
            s = f"{self.conv_layer_name}"
        else:
            s = ""
        if self.threshold is not None and self.threshold > 0.0 and not self.threshold_mean_plus_std:
            s += f"_threshold_{self.threshold}"
        elif self.threshold is not None and self.threshold == 1 and self.threshold_mean_plus_std:
            s += f"_thr_mean_plus_std"
        elif self.threshold is not None and self.threshold >= 0.0 and self.threshold_mean_plus_std:
            s += f"_thr_mean_plus_std_{self.threshold}"
        s += f"_{self.vgg_depth}"
        return s