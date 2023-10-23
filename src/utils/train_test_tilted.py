"""
Train and test functionality 
"""

from tqdm import tqdm
import numpy as np
from functools import partial
from time import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regularizer import tilted_loss, l1_loss
from .augmentation import get_noisy_images

from deepillusion.torchattacks import PGD

import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import os



def single_epoch_tilted(cfg,
                 model,
                 train_loader,
                 optimizer,
                 scheduler=None,
                 logging_func=print,
                 verbose: bool = True,
                 epoch: int = 0):
    r"""
    Single epoch for the titled loss function
    """
    start_time = time()
    model.train()
    device = model.parameters().__next__().device

    cross_ent = nn.CrossEntropyLoss()
    train_loss = 0
    train_correct = 0
    
    schedule_tilt_costs = True


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        if "tilt" in cfg.train.regularizer.active:
            _ = model(data)


            tilt_loss_per_layer: list[torch.Tensor] = []
            for idx, (layer, layer_input, layer_output) in enumerate(zip(model.layers_of_interest.values(), model.layer_inputs.values(), model.layer_outputs.values())):
                 
                if idx < cfg.nn.num_texp_layers:                  

                    loss_args = dict(activations=layer_output,
                                    tilt=cfg.train.regularizer.tilt.tilt[idx]*cfg.nn.norm_layer_tilt[idx], 
                                    relu_on=cfg.reg_relu_on,
                                    anti_hebb=cfg.reg_anti_hebbian,
                                    texp_across_filts_only=cfg.texp_across_filts_only)
                             
                    tilted_loss_ = tilted_loss(**loss_args)

                    tilt_loss_per_layer.append(tilted_loss_)    
                   
                else:
                    tilt_loss_per_layer.append(torch.zeros(1))  



            if "tilt" in cfg.train.regularizer.active:
                scalar_vector = torch.Tensor(cfg.train.regularizer.tilt.alpha).to(device)

                if schedule_tilt_costs:
                # Introduce TEXP cost after the filters have had time to settle down. This is not necessary, produces no significant difference in the performance of the learnt models.
                    if epoch < 5:
                        scalar_vector = torch.zeros_like(scalar_vector)

                # The following method of coding gives flexibility to extend TEXP to multiple layers, and ensure that the TEXP cost for a particular layer is tied only to that specific layer's weights.
                for idx, (layer, _, _) in list(enumerate(zip(model.layers_of_interest.values(), model.layer_inputs.values(), model.layer_outputs.values())))[::-1]:
                    if scalar_vector[idx] == 0.0:
                        continue
                    if layer.weight.grad is not None:
                        layer.weight.grad.zero_()

                    layer_tilt_loss = -scalar_vector[idx] * \
                        tilt_loss_per_layer[idx]
                    layer_tilt_loss.backward(retain_graph=True)



        # Noisy Training
        if cfg.train.type == "noisy":
            data = get_noisy_images(data, cfg.train.noise.std)

        # Adversarial Training
        if cfg.train.type == "adversarial":           
            perturbs = PGD(net=model, x=data, y_true=target, data_params={
                "x_min": cfg.dataset.min, "x_max": cfg.dataset.max}, attack_params=cfg.train.adversarial, verbose=False)
           
            data += perturbs


        loss = 0.0
        output = model(data)
        xent_loss = cross_ent(output, target)

        l1_weight_loss = 0
        if "l1_weight" in cfg.train.regularizer.active:
            for _, layer in model.named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    l1_weight_loss += l1_loss(
                        features={"conv": layer.weight}, dim=(1, 2, 3))

        loss += xent_loss + cfg.train.regularizer.l1_weight.scale * l1_weight_loss

        loss.backward()
        optimizer.step()
        if scheduler and cfg.nn.scheduler == "cyc":
            scheduler.step()

        train_loss += xent_loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=False)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()


    if scheduler and not cfg.nn.scheduler == "cyc":
        scheduler.step()
    
    train_size = len(train_loader.dataset)
    train_loss = train_loss/train_size
    train_acc = train_correct/train_size

   
    
    if verbose:
        logging_func(
            f"Epoch: \t {epoch} \t Time (s): {(time()-start_time):.2f}")
        logging_func(
            f"Train Xent loss: \t {train_loss:.4f} \t Train acc: {100*train_acc:.2f} %")
        logging_func(f"L1 Weight Loss: \t {l1_weight_loss:.4f}")
        
        if "tilt" in cfg.train.regularizer.active:
            layerwise_tilt_loss = []
            scaled_layerwise_tilt_loss = []
            for idx2, loss_val in enumerate(tilt_loss_per_layer):
                layerwise_tilt_loss.append(loss_val.cpu().detach().numpy().item())
                scaled_layerwise_tilt_loss.append(loss_val.cpu().detach().numpy().item()*scalar_vector[idx2].cpu().detach().numpy().item())
            
            logging_func(
                f"TEXP objective per layer (to be maximized): {layerwise_tilt_loss:}")
            logging_func(
                f"Weighted TEXP obj per layer (to be maximized): {scaled_layerwise_tilt_loss:}")

        logging_func("-"*100)
    

