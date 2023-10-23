"""
"""

from tqdm import tqdm
import numpy as np
from functools import partial
from time import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regularizer import l1_loss
from .augmentation import get_noisy_images


from deepillusion.torchattacks import PGD, FGSM, RFGSM, BIM, PGD_EOT
from deepillusion.torchattacks.analysis import whitebox_test, substitute_test
from deepillusion.torchattacks.analysis import get_perturbation_stats

from robustbench.data import load_cifar10, load_cifar10c, _load_dataset




def standard_test(model, test_loader, verbose=True, progress_bar=False):
    """
    Description: Evaluate model with test dataset,
        if adversarial args are present then adversarially perturbed test set.
    Input :
        model : Neural Network               (torch.nn.Module)
        test_loader : Data loader            (torch.utils.data.DataLoader)
        verbose: Verbosity                   (Bool)
        progress_bar: Progress bar           (Bool)
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    device = model.parameters().__next__().device
    start_time = time()
    model.eval()

    test_loss = 0
    test_correct = 0

    if progress_bar:
        iter_test_loader = tqdm(
            iterable=test_loader,
            unit="batch",
            leave=False)
    else:
        iter_test_loader = test_loader

    test_size = len(test_loader.dataset)

    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)

        output = model(data)
        
        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=False)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"\t Standard Test: Time (s): {(time()-start_time):.2f}")

    if verbose:
        print(
            f"Test loss: {test_loss/test_size:.4f}, Test acc: {100*test_correct/test_size:.2f}")


    return test_loss/test_size, test_correct/test_size


def adversarial_test(model, test_loader, adversarial_args, verbose=True, progress_bar=False):
    """
    Description: Evaluate model with test dataset with adversarial perturbations
    Input :
        model : Neural Network               (torch.nn.Module)
        test_loader : Data loader            (torch.utils.data.DataLoader)
        adversarial_args :
        verbose: Verbosity                   (Bool)
        progress_bar: Progress bar           (Bool)
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    device = model.parameters().__next__().device

    model.eval()

    test_loss = 0
    test_correct = 0
    test_correct_top5 = 0

    if progress_bar:
        iter_test_loader = tqdm(
            iterable=test_loader,
            unit="batch",
            leave=False)
    else:
        iter_test_loader = test_loader

    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)

        # Perturb the data with PGD attack
        pgd_args = dict(net=model,
                        x=data,
                        y_true=target,
                        data_params=adversarial_args["attack_args"]["data_params"],
                        attack_params=adversarial_args["attack_args"]["attack_params"],
                        verbose=False,
                        progress_bar=False)               
        perturbs = PGD(**pgd_args)
        data_adversarial = data + perturbs
        output = model(data_adversarial)
        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data_adversarial.size(0)

        pred = output.argmax(dim=1, keepdim=False)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

        test_correct_top5 += target.unsqueeze(dim=1).eq(torch.topk(output, k=5, dim=1)[1]).sum().item()

    test_size = len(test_loader.dataset)
    if verbose:
        print(
            f"Test loss: {test_loss/test_size:.4f}, Test acc: {100*test_correct/test_size:.2f}, Test acc top-5: {100*test_correct_top5/test_size:.2f}")

    return test_loss/test_size, test_correct/test_size, test_correct_top5/test_size


def common_corruptions_test(x_orig, y_orig, cfg, device, model, verbose=False):    
    device = model.parameters().__next__().device

    model.eval()
    bs = cfg.test.batch_size
    n_batches = math.ceil(x_orig.shape[0] / bs)
    test_loss = 0
    test_correct = 0
    test_correct_top5 = 0

    for counter in range(n_batches):

        data = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        target = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target.type(torch.LongTensor).to(device)).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=False)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

        test_correct_top5 += target.unsqueeze(dim=1).eq(torch.topk(output, k=5, dim=1)[1]).sum().item()

    test_size = x_orig.shape[0]

    if verbose:
        print(
            f"Corruption \t Test loss: {test_loss/test_size:.4f}, Test acc: {100*test_correct/test_size:.2f}, Test acc top5: {100*test_correct_top5/test_size:.2f}")

    return test_loss/test_size, test_correct/test_size, test_correct_top5/test_size




def test_common_corruptions(cfg, device, model, logger): 
    # Evaluate on common corruptions datatset:
    corruptions = ('gaussian_noise', 'shot_noise',  'impulse_noise','speckle_noise', 'snow', 'frost', 'fog', 'brightness',  'spatter','defocus_blur',  'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur','contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',  'saturate',)
    all_severities = True 
    test_acc_all = []
    test_acc_top5_all = []
    test_acc_sev5_all = []
    test_acc_sev5_top5_all = []

    for corr_idx in range(len(corruptions)):
        corruption = (corruptions[corr_idx],)
        if cfg.dataset.name == 'CIFAR10':
            x_test, y_test = load_cifar10c(n_examples=10000, data_dir=cfg.dataset.directory, corruptions=corruption, all_severities=all_severities)
            x_test_sev5, y_test_sev5 = load_cifar10c(n_examples=50000, data_dir=cfg.dataset.directory, corruptions=corruption)
        else:
            raise NotImplementedError

        test_loss_corr, test_acc_corr, test_acc_top5_corr = common_corruptions_test(x_test, y_test, cfg, device, model)
        test_loss_corr_sev5, test_acc_corr_sev5, test_acc_top5_corr_sev5 = common_corruptions_test(x_test_sev5, y_test_sev5, cfg, device, model)

        test_acc_all.append(test_acc_corr)
        test_acc_sev5_all.append(test_acc_corr_sev5)

        if cfg.dataset.name != 'CIFAR10':
            test_acc_top5_all.append(test_acc_top5_corr)
            test_acc_sev5_top5_all.append(test_acc_top5_corr_sev5)
        
        logger.info(f'Corruption: {corruption[0]} \t Test \t acc all: {test_acc_corr:.4f} \t acc sev5: {test_acc_corr_sev5:.4f} \t Top-5 acc all: {test_acc_top5_corr:.4f} \t Top-5 acc sev5: {test_acc_top5_corr_sev5:.4f}')
    
    logger.info(f'Accuracy on all corruptions: {([val*100 for val in test_acc_all])}')
    logger.info(f'Average accuracy on all corruptions: {(100*sum(test_acc_all)/len(test_acc_all))}')
    logger.info(f'Min accuracy among all corruptions: {(100*min(test_acc_all))}')
    logger.info(f'Max accuracy among all corruptions: {(100*max(test_acc_all))}')

       
    if all_severities:
        logger.info(f'The above accuracies are averaged over all severities of each corruption. Below accuracies are for the highest severity of each corruption.')

    logger.info(f'Severity level 5: Accuracy on all corruptions: {([val*100 for val in test_acc_sev5_all])}')
    logger.info(f'Severity level 5: Average accuracy on all corruptions: {(100*sum(test_acc_sev5_all)/len(test_acc_sev5_all))}')
    logger.info(f'Severity level 5: Min accuracy among all corruptions: {(100*min(test_acc_sev5_all))}')
    logger.info(f'Severity level 5: Max accuracy among all corruptions: {(100*max(test_acc_sev5_all))}')



def test_snr(model, test_loader, noise_std):

    device = model.parameters().__next__().device
    model.eval()
    sum_snr = torch.Tensor(torch.zeros(len(model.layers_of_interest)))


    for data, _ in test_loader:

        data = data.to(device)
        noisy_data = get_noisy_images(data, noise_std, "gaussian")

        sig = [[] for x in range(len(model.layers_of_interest))]
        noisy_sig = [[] for x in range(len(model.layers_of_interest))]

        _ = model(data)
        for layer_idx, layer in enumerate(model.layers_of_interest.keys()):
            sig[layer_idx] = model.layer_inputs[layer]
        
        _ = model(noisy_data)
        for layer_idx, layer in enumerate(model.layers_of_interest.keys()):
            noisy_sig[layer_idx] = model.layer_inputs[layer]

        for layer_idx in range(len(sig)):
            sum_snr[layer_idx] = sum_snr[layer_idx] + torch.sum(torch.square(torch.linalg.vector_norm(sig[layer_idx],ord=2,dim=(1,2,3))) / torch.square(torch.linalg.vector_norm((sig[layer_idx] - noisy_sig[layer_idx]),ord=2,dim=(1,2,3))))
               

    test_size = len(test_loader.dataset)
    exp_snr_dataset = 10*np.log10(sum_snr/test_size)
    return exp_snr_dataset 


