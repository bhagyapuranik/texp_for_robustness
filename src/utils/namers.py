import torch
import os
import numpy as np

from omegaconf import DictConfig


def classifier_params_string(model_name: str, cfg: DictConfig):
    classifier_params_string = model_name

    classifier_params_string += f"_lr_{cfg.nn.lr:.4f}"

    if cfg.train.type == "noisy":
        classifier_params_string += f"_noisytr_{'_'.join([str(x) for x in cfg.train.noise.values()])}"

    if cfg.train.type == "adversarial":
        adv_name_string = ""
        for x in cfg.train.adversarial.values():
            if x != "cross_entropy":
                adv_name_string += f"_{str(x)}"
        classifier_params_string += f"_advtr{adv_name_string}"

    if "l1_weight" in cfg.train.regularizer.active:
        classifier_params_string += f"_l1_weight_{'_'.join([str(x) for x in cfg.train.regularizer.l1_weight.values()])}"




    if "tilt" in cfg.train.regularizer.active:
        classifier_params_string += f"_tilt"
        classifier_params_string += f"_{'_'.join([str(x) for x in cfg.train.regularizer.tilt.values()])}"
        if cfg.nn.norm_layer_type == "texp":
            classifier_params_string += f"_normTilt_{cfg.nn.norm_layer_tilt}"
        if cfg.texp_across_filts_only:
            classifier_params_string += f"_filt"

    if cfg.dataset.name != "CIFAR10":
        classifier_params_string += f"_{cfg.dataset.name}"

    if cfg.train.sota_augs!="none":
        classifier_params_string += f"_{cfg.train.sota_augs}"

    classifier_params_string += f"_ep_{cfg.train.epochs}"
    classifier_params_string += f"_seed_{cfg.seed}"


    return classifier_params_string


def classifier_ckpt_namer(model_name: str, cfg: DictConfig):

    file_path = cfg.directory + f"checkpoints/{cfg.dataset.name}/"
    os.makedirs(file_path, exist_ok=True)

    file_path += classifier_params_string(model_name, cfg)

    file_path += ".pt"

    return file_path


def classifier_log_namer(model_name: str, cfg: DictConfig):

    file_path = cfg.directory + f"logs/{cfg.dataset.name}/"

    os.makedirs(file_path, exist_ok=True)

    file_path += classifier_params_string(model_name, cfg)

    file_path += ".log"

    return file_path
