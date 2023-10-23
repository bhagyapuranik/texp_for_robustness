#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

COMMAND="python -m src.train_model --multirun
                train.type=standard
                nn.classifier=TEXP_VGG
                nn.conv_layer_type=implicitconv2d
                nn.norm_layer_type=texp    
                nn.norm_layer_tilt=[0.192]
                nn.threshold=0.5
                nn.threshold_mean_plus_std=true
                texp_across_filts_only=true
                nn.lr=0.001
                train.epochs=100
                train.regularizer.l1_weight.scale=0.001
                train.regularizer.active=['tilt']
                train.regularizer.tilt.tilt=[10]
                nn.texp_count=0
                nn.num_texp_layers=1
                train.regularizer.tilt.alpha=[0.001,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                dataset.name='CIFAR10'
                train.batch_size=128
                test.batch_size=100
                save_model=true
                train.sota_augs=none"

echo $COMMAND
eval $COMMAND
