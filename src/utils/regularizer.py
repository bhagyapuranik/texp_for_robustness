from optparse import Option
from typing import Dict, Union, Tuple, Optional
#from cv2 import threshold
import numpy as np
from math import ceil
from requests import patch
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["l1_loss", "tilted_loss"]


def l1_loss(features: Dict[str, torch.Tensor] = {}, dim: Union[int, Tuple[int]] = None) -> torch.float:
    """
    L1 loss: L1 norm of the given tensors disctionary

    Takes a dictionary of tensors

    returns |feature|_1
    """
    loss = 0
    for feature in features:
        loss += torch.mean(torch.sum(torch.abs(features[feature]), dim=dim))
    return loss


def tilted_loss(activations: torch.Tensor, tilt: float, dim: int = 1, 
                relu_on=True, anti_hebb=False, texp_across_filts_only=False, **kwargs):
    if relu_on:
        activations = F.relu(activations)   
    
    if anti_hebb:
        mean_subtraction = True # Balanced TEXP, subtracts the mean of the activations before exponentiating
    else:
        mean_subtraction = False

    if mean_subtraction:
        if texp_across_filts_only:
            mean_acts = torch.mean(activations, dim=(1), keepdim=True)
        else:
            mean_acts = torch.mean(activations, dim=(1,2,3), keepdim=True)
    else:
        mean_acts = torch.zeros_like(activations)


    if texp_across_filts_only:
        return (1/(activations.shape[2]*activations.shape[3]))*(1/tilt)*torch.add( torch.sum(torch.logsumexp(tilt*(activations-mean_acts), dim=(1))), (activations.nelement()/activations.shape[1])*math.log(1/activations.shape[1]) )
    else:
        return (1/tilt)*torch.add( torch.sum(torch.logsumexp(tilt*(activations-mean_acts), dim=(1,2,3))), activations.shape[0]*math.log(activations.shape[0]/activations.nelement()) )


    