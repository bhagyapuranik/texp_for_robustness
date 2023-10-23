from typing import Dict, Iterable, Callable
import numpy as np
from tqdm import tqdm

from functools import partial
from .layers import AdaptiveThreshold

import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
from torch import nn
import os

__all__ = ["count_parameter"]


def count_parameter(model: torch.nn.Module, logger: Callable = print, verbose: bool = True) -> int:
    """
    Outputs the number of trainable parameters on the model
    """

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    if verbose:
        logger(f" Number of total trainable parameters: {params}")
    return params

