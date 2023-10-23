import torch
import torch.nn as nn
from torchvision import datasets, transforms

import numpy as np
from os import path
import PIL


def cifar10(cfg):

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    if cfg.train.sota_augs == "none":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ]
            )
    elif cfg.train.sota_augs == "randaug":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(),
                transforms.ToTensor(),
                ]
            )
    elif cfg.train.sota_augs == "augmix":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AugMix(),
                transforms.ToTensor(),
                ]
            )
    elif cfg.train.sota_augs == "autoaug":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                ]
            )
        
    transform_test = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.CIFAR10(
        root=cfg.dataset.directory,
        train=True,
        download=True,
        transform=transform_train,
        )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=2
        )

    testset = datasets.CIFAR10(
        root=cfg.dataset.directory,
        train=False,
        download=True,
        transform=transform_test,
        )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.test.batch_size, shuffle=False, num_workers=2
        )

    return train_loader, test_loader


def cifar100(cfg):

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    if cfg.train.sota_augs == "none":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ]
            )
    elif cfg.train.sota_augs == "randaug":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(),
                transforms.ToTensor(),
                ]
            )
    elif cfg.train.sota_augs == "augmix":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AugMix(),
                transforms.ToTensor(),
                ]
            )
    elif cfg.train.sota_augs == "autoaug":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(AutoAugmentPolicy = transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                ]
            )

    transform_test = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.CIFAR100(
        root=cfg.dataset.directory,
        train=True,
        download=True,
        transform=transform_train,
        )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4
        )

    testset = datasets.CIFAR100(
        root=cfg.dataset.directory,
        train=False,
        download=True,
        transform=transform_test,
        )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.test.batch_size, shuffle=False, num_workers=4
        )

    return train_loader, test_loader


def get_loaders(cfg):

    if cfg.dataset.name == "CIFAR10":
        train_loader, test_loader = cifar10(cfg)
    else:
        raise NotImplementedError

    return train_loader, test_loader
