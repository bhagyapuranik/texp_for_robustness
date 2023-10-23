"""

"""

import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import os

import torch
from torch.nn import Conv2d


# PYTORCH UTILS
from .utils import SpecificLayerTypeOutputExtractor_wrapper, count_parameter, standard_test, test_noisy, test_common_corruptions
from .utils.namers import classifier_ckpt_namer
from .utils.train_test_tilted import single_epoch_tilted


# ATTACK CODES
from robustbench.data import load_cifar10, load_cifar10c, _load_dataset
from autoattack import AutoAttack

# Initializers
from .init import *


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), verbose=True)
project_dir = os.getenv("PROJECT_DIR")


@hydra.main(config_path=project_dir + "src/configs", config_name="cifar")
def main(cfg: DictConfig) -> None:
    cfg.directory = project_dir

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    extra_tests_on = False # Set to true to perform the complete set of robustness evaluations reported in the paper.

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader, _ = init_dataset(cfg)
    
    model = init_classifier(cfg).to(device)
    model = SpecificLayerTypeOutputExtractor_wrapper(
        model=model, layer_type=globals()[cfg.train.regularizer.tilt.layer])

    logger = init_logger(cfg, model.name)

    if cfg.verbose:
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(model)
        logger.info(f"Model will be saved to {classifier_ckpt_namer(model_name=model.name, cfg=cfg)}")

   
    optimizer, scheduler = init_optimizer_scheduler(cfg, model, len(
        train_loader), printer=logger.info, verbose=cfg.verbose)
    _ = count_parameter(model=model, logger=logger.info, verbose=cfg.verbose)



    for epoch in range(1, cfg.train.epochs+1):
        single_epoch_tilted(cfg=cfg, model=model, train_loader=train_loader,
                        optimizer=optimizer, scheduler=scheduler, verbose=True, epoch=epoch)
        if epoch % cfg.log_interval == 0 or epoch == cfg.train.epochs:
            _, _, = standard_test(model=model, test_loader=test_loader,
                                  verbose=True, progress_bar=False)
        if cfg.save_model:
            os.makedirs(cfg.directory + "checkpoints/classifiers/", exist_ok=True)
            classifier_filepath = classifier_ckpt_namer(model_name=model.name, cfg=cfg)
            torch.save(model.state_dict(), classifier_filepath)
            logger.info(f"Saved to {classifier_filepath}")

    if extra_tests_on:
        noisy_acc = [None]*cfg.noisy.num_experiments
        noisy_acc_top5 = [None]*cfg.noisy.num_experiments
        noisy_loss = [None]*cfg.noisy.num_experiments
        for i in range(cfg.noisy.num_experiments):
            noise_std = cfg.noisy.std
            noisy_acc[i], noisy_acc_top5[i], noisy_loss[i], _ = test_noisy(model, test_loader, noise_std)
        logger.info(f'Noise std {noise_std:.4f}: Test  \t loss: {sum(noisy_loss)/cfg.noisy.num_experiments:.4f} \t acc: {sum(noisy_acc)/cfg.noisy.num_experiments:.4f} \t top5 acc: {sum(noisy_acc_top5)/cfg.noisy.num_experiments:.4f}')

    
        test_common_corruptions(cfg, device, model, logger)

        
        # Auto Attack
        # L-2 test
        if cfg.dataset.name == 'CIFAR10':
            x_test, y_test = load_cifar10(n_examples=10000, data_dir=cfg.dataset.directory)
        else:
            raise NotImplementedError    

        adversary = AutoAttack(model, norm='L2', eps=cfg.test.l2_eps, version="standard",
                            seed = cfg.seed)        
        adversary.apgd.n_restarts = 1

        print("#" + "-"*100 + "#")
        print("-"*40 + " L_2 " + "-"*40)
        print("#" + "-"*100 + "#")
        logger.info("L_2 attack eps: " + str(cfg.test.l2_eps))
        x_adv = adversary.run_standard_evaluation(x_test, y_test)

        # L-inf test
        adversary = AutoAttack(model, norm='Linf', eps=cfg.test.linf_eps, version="standard",
                             seed = cfg.seed)
        adversary.apgd.n_restarts = 1

        print("#" + "-"*100 + "#")
        print("-"*40 + " L_inf " + "-"*40)
        print("#" + "-"*100 + "#")
        logger.info("L_inf attack eps: " + str(cfg.test.linf_eps))
        x_adv = adversary.run_standard_evaluation(x_test, y_test)
    

if __name__ == "__main__":
    main()
