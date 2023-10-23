"""

"""
from torch.hub import load_state_dict_from_url
from cgi import test
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
import os
import math

# ATTACK CODES
from robustbench.data import load_cifar10, load_cifar10c, _load_dataset
from autoattack import AutoAttack

# Initializers
from .init import *

from .utils import standard_test, test_noisy, test_common_corruptions, test_snr, SpecificLayerTypeOutputExtractor_wrapper
from .utils.namers import classifier_ckpt_namer
from .utils import settings
from .utils.analysis import *

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), verbose=True)
project_dir = os.getenv("PROJECT_DIR")

@hydra.main(config_path=project_dir + "src/configs", config_name="cifar")
def main(cfg: DictConfig) -> None:

    cfg.directory = project_dir

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    _, test_loader, __ = init_dataset(cfg)

    model = init_classifier(cfg).to(device)

    logger = init_logger(cfg, model.name)

    classifier_filepath = classifier_ckpt_namer(model_name=model.name, cfg=cfg)
    model.load_state_dict(torch.load(classifier_filepath))
    model = SpecificLayerTypeOutputExtractor_wrapper(model=model, layer_type=torch.nn.Conv2d)
    model.eval()    
    
    for p in model.parameters():
        p.requires_grad = False



    test_loss, test_acc = standard_test(
        model=model, test_loader=test_loader, verbose=False, progress_bar=False)
    logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

    
    
    # Noisy Test Set Evaluation
    noisy_acc = [None]*cfg.noisy.num_experiments
    noisy_acc_top5 = [None]*cfg.noisy.num_experiments
    noisy_loss = [None]*cfg.noisy.num_experiments
    for i in range(cfg.noisy.num_experiments):
        noise_std = cfg.noisy.std
        noisy_acc[i], noisy_acc_top5[i], noisy_loss[i], _ = test_noisy(model, test_loader, noise_std)

    logger.info(f'Noise std {noise_std:.4f}: Test  \t loss: {sum(noisy_loss)/cfg.noisy.num_experiments:.4f} \t acc: {sum(noisy_acc)/cfg.noisy.num_experiments:.4f} \t top5 acc: {sum(noisy_acc_top5)/cfg.noisy.num_experiments:.4f}')


    # Evaluate on common corruptions datatset:
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




