#!/bin/bash

["BatchSize", "SGD_lr", "WD", "Mom" "SWA_lr", "K"]
batch_sizes=(64 128)
sgd_lrs=(0.05 0.1)
weight_decays=(3e-4 5e-4)
swa_lrs=(0.01 0.05)

for BS in ${batch_sizes[@]}; do
    for SGD_LR in ${sgd_lrs[@]}; do
        for WD in ${weight_decays[@]}; do
            for SWA_LR in ${swa_lrs[@]}; do
                dir="./tuning/${BS}_${SGD_LR}_${WD}_${SWA_LR}"
                in_file="${dir}/swag-20.pt"
                out_file="${dir}/val_eval"
                python -u swag_modified/uncertainty/uncertainty.py --file=$in_file \
                        --data_path="./data" --train_dataset="uuss_train.h5" --validation_dataset="uuss_validation.h5" \
                        --n_duplicates_train=3 --batch_size=${BS} --method=SWAG --cov_mat --scale=0.5 \
                        --save_path=$out_file
                wait -n
                wait
            done
        done
    done
done