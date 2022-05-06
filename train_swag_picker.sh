#!/bin/bash

epochs=20
swa_start=10
save_freq=5

# -----------  --------  --------  --------  --------  ---
#   BatchSize    SGD_lr        WD       Mom    SWA_lr    K
# -----------  --------  --------  --------  --------  ---
#         128    0.0500    0.0003    0.9000    0.0100   20
        
batch_size=128
sgd_lr=0.05
weight_decay=3e-4
swa_lr=0.01
seeds=(2 3)

for seed in ${seeds[@]}; do
        dir="./ensembles/seed${seed}_${batch_size}_${sgd_lr}_${weight_decay}_${swa_lr}"
        echo $dir
        python -u swag_modified/train/run_swag.py --data_path="./data" \
                --train_dataset="uuss_train.h5" --validation_dataset="uuss_validation.h5" \
                --load_model="existing_models/SCSN_model_004.pt" --n_duplicates_train=3 \
                --batch_size=${batch_size} --epochs=${epochs} --model="PPicker" --save_freq=${save_freq} --lr_init=${sgd_lr} \
                --wd=${weight_decay} --swa --swa_start=${swa_start} --swa_lr=${swa_lr} --cov_mat --dir=${dir} --seed=${seed}
        wait -n 
        wait
done
