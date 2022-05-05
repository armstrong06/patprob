#!/bin/bash

python -u swag_modified/uncertainty/uncertainty.py --file="tuning/128_0.05_3e-4_0.01/swag-20.pt" \
        --data_path="./data" --train_dataset="uuss_train.h5" --validation_dataset="uuss_test_fewerhist.h5" \
        --n_duplicates_train=3 --batch_size=128 --method=SWAG --cov_mat --scale=0.5 \
        --save_path="tuning/128_0.05_3e-4_0.01/swag_test_uncertainty"
