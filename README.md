# P Arrival Time Probability 
## A Bayesian approach for estimating P-arrival times and their uncertainty. 
### Alysha Armstrong - Class project for CS 6190

## Overview
Uses the model architecture presented by Ross et al.(2018) to adjust an estimated P-arrival time to emulate the results of expert seismic analysts. The Bayesian approximation of the P-arrival time pick is determined by incoporating Stochastic Model Averging - Gaussian (SWAG) (Maddox et al., 2019) into the training and evaluation of the Ross architecture. 

## File Structure 
+-- data/ (University of Utah Seismograph Stations vertical component waveform data)
+-- ensembles/ 
|   +-- figures/ (figures for the ensemble predictions on the test dataset)
    +-- seed1_128_0.05_3e-4_0.01/ (Ensemble model that uses a random seed of 1)
    +-- seed2_128_0.05_3e-4_0.01/ (Ensemble model that uses a random seed of 2)
    +-- seed3_128_0.05_3e-4_0.01/ (Ensemble model that uses a random seed of 3)
    +--ensemble_results.py (Use to combine results from various models and analyze)
+-- existing_models/ (pre-trained models that use a larger dataset)
+-- log_files/ (log files from running various scripts)
+-- swa_gaussian-master/ (original SWAG code from https://github.com/wjmaddox/swa_gaussian)
+-- swag_modifies/ (modified SWAG code to work for this problem)
|   +-- swag/
    |   +-- models/ (Folder with the model architectures)
    |   +-- posteriors/ (Folder that contains SWAG model definition)
    +-- losses.py (various loss functions)
    +-- seismic_data.py (script for processing the seismic data for training & evaluation)
    +-- utils.py (Various functions for training, testing, making predictions, etc.)
+-- train_results_practice/ (Folder contatining training and evalution results for the first model I ran, trying to get things to work)
+-- tuning/ (Folder containing the models and results from hyperparameter tuning)
