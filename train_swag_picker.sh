#!/bin/bash

python -u swag_modified/train/run_swag.py --data_path="./data" \
        --train_dataset="uuss_train.h5" --validation_dataset="uuss_validation.h5" \
        --load_model="existing_models/SCSN_model_004.pt" --n_duplicates_train=3 \
        --batch_size=64 --epochs=60 --model="PPicker" --save_freq=30 --lr_init=0.05 \
        --wd=5e-4 --swa --swa_start=31 --swa_lr=0.01 --cov_mat --dir="./train_results"