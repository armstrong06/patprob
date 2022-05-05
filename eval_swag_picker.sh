#!/bin/bash

python -u swag_modified/uncertainty/uncertainty.py --file="./train_results/swag-60.pt" \
        --data_path="./data" --train_dataset="uuss_train.h5" --validation_dataset="uuss_test_fewerhist.h5" \
        --n_duplicates_train=3 --batch_size=64 --method=SWAG --cov_mat --scale=0.5 \
        --save_path="./train_results/swag_uncertainty"
