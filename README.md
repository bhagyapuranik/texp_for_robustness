# TEXP approach for enhancing robustness of DNNs

## Simplified models:

The code to create simplified data models and apply unsupervised TEXP learning on a single layer neural network is contained in the jupyter notebook simplified_models.ipynb


## Pre-requisites for DNN codebase

Dependencies are listed in the file requirements.txt. Packages like robustbench and autoattack should be installed by cloning the original repositories at https://github.com/RobustBench/robustbench and https://github.com/fra31/auto-attack. Autoattack, robustbench are only needed for evaluation. All the results on VGG based models with the CIFAR-10 dataset were obtained using this codebase.

To install the requirements:
```bash
pip install -r requirements.txt
```

## Creating a .env file:

After cloning the repository, enter the project folder and create .env file. Then add the current directory (project directory) to .env file as:

```bash
PROJECT_DIR=<project directory>/texp_for_robustness/
```


## Hyperparameters

All the hyperparameters and other settings are located inside the file src/configs/cifar.yaml. Some of the parameters are exposed in the shell scripts to train and evaluate models, where settings from the config file can be overriden.


## Training

The default parameters and settings to train a TEXP-VGG-16 model are loaded in src/sh/train.sh. To launch the training, execute the following from the project directory. It also saves the checkpoint.

```bash
bash src/sh/train.sh
```

## Evaluation

Evaluate the model using the trained checkpoint. The parameters in eval.sh are set to match the default parameters in train.sh.

```bash
bash src/sh/eval.sh
```

## Note 1:
Common corruptions evaluation function test_common_corruptions() relies upon loader functions inside robustbench --> data.py --> load_cifar10c() which have been modified to return corruptions of all severity levels, when needed. The file is located at robustbench_helper/data.py. After installing robustbench, please replace the original file at "site-packages/robustbench/data.py" with this version and appropriately modify the enums.

## Note 2:
Tests on CIFAR-100 with WRN-28-10 and on ImageNet with ResNet-50 were performed by cloning publicly available repositories on baseline WRN and ResNet training, and adding a single layer of TEXP blocks on top of it. Details and acknowledgements will be released later.




