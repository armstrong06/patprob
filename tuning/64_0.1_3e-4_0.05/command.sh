swag_modified/train/run_swag.py --data_path=./data --train_dataset=uuss_train.h5 --validation_dataset=uuss_validation.h5 --load_model=existing_models/SCSN_model_004.pt --n_duplicates_train=3 --batch_size=64 --epochs=20 --model=PPicker --save_freq=5 --lr_init=0.1 --wd=3e-4 --swa --swa_start=10 --swa_lr=0.05 --cov_mat --dir=./tuning/64_0.1_3e-4_0.05
