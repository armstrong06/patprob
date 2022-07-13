#!/bin/bash

# python -u swag_modified/uncertainty/uncertainty.py --file="tuning/128_0.05_3e-4_0.01/swag-20.pt" \
#         --data_path="./data" --train_dataset="uuss_train.h5" --validation_dataset="uuss_test_fewerhist.h5" \
#         --n_duplicates_train=3 --batch_size=128 --method=SWAG --cov_mat --scale=0.5 \
#         --save_path="tuning/128_0.05_3e-4_0.01/swag_test_uncertainty"

epochs=20
# swa_start=10
# save_freq=5

# -----------  --------  --------  --------  --------  ---
#   BatchSize    SGD_lr        WD       Mom    SWA_lr    K
# -----------  --------  --------  --------  --------  ---
#         128    0.0500    0.0003    0.9000    0.0100   20
        
batch_size=128
sgd_lr=0.05
weight_decay=3e-4
swa_lr=0.01
seeds=(1 2 3)

for seed in ${seeds[@]}; do
        dir="./ensembles/seed${seed}_${batch_size}_${sgd_lr}_${weight_decay}_${swa_lr}"
        echo $dir
        python -u swag_modified/uncertainty/uncertainty.py --file="${dir}/swag-${epochs}.pt" \
                --data_path="./data" --train_dataset="uuss_train.h5" --validation_dataset="uuss_validation.h5" \
                --n_duplicates_train=3 --batch_size=${batch_size} --method="SWAG" --cov_mat --scale=0.5 --seed=${seed}\
                --save_path="${dir}/swag_validation_uncertainty"
        wait -n 
        wait
done
