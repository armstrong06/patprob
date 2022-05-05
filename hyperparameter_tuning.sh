#!/bin/bash

# ["BatchSize", "SGD_lr", "WD", "Mom" "SWA_lr", "K"]
batch_sizes=(64, 128)
sgd_lrs=(0.05, 0.1)
weight_decays=(3e-4, 5e-4)
swa_lrs=(0.01, 0.05)

for BS in ${batch_sizes[@]}; do
    for SGD_LR in ${sgd_lrs[@]}; do
        for WD in ${weight_decays[@]}; do
            for SWA_LR in ${swa_lrs[@]}; do
                python -u swag_modified/train/run_swag.py --data_path="./data" \
                        --train_dataset="uuss_train.h5" --validation_dataset="uuss_validation.h5" \
                        --load_model="existing_models/SCSN_model_004.pt" --n_duplicates_train=3 \
                        --batch_size=$BS --epochs=30 --model="PPicker" --save_freq=5 --lr_init=$SGD_LR \
                        --wd=$WD --swa --swa_start=16 --swa_lr=$SWA_LR --cov_mat --dir="./hyperparams/${BS}_${SGD_LR}_${WD}_${SWA_LR}"