import numpy as np
import torch
import os
import sys
import h5py 

class NumpyDataset(torch.utils.data.Dataset):

    def __init__(self, data, target, transform=None):
        n_obs = data.shape[0]
        n_samples = data.shape[1]
        self.data = torch.from_numpy(data.reshape([n_obs, 1, n_samples])).float()
        self.target = torch.from_numpy(target.reshape([n_obs, 1])).float()

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)

def randomize_start_times_and_normalize(X_in, time_series_len = 400,
                                        max_dt=0.5, dt=0.01,
                                        n_duplicate = 1):
    """
    Uniformly randomize's the start times to within +/- max_dt seconds of
    the trace center and normalizes the amplitudes to [-1,1].
   
    Parameter
    ---------
    X_in : np.ndarray
       The n_obs x n_samples matrix of observed time series.  By this point
       all observations have been uniformly resampled.
    time_series_len : integer
       This is the desired output length.  This cannot exceed X_in.shape[1].
    max_dt : double
       The maximum time lag in seconds.  For example, 0.5 means the traces
       can be shifted +/- 0.5 seconds about the input trace center.
    dt : double
       The sampling period in seconds.

    Returns
    -------
    X_out : np.ndarray
       The n_obs x time_series_len matrix of signals that were normalized to 
       +/- 1 and whose start times were randomizes to +/- max_dt of their
       original start time.
    random_lag : np.array
       The time shift to add to the original pick time to obtain the new
       pick time. 
    """
    max_shift = int(max_dt/dt)
    n_obs = X_in.shape[0] 
    n_samples = X_in.shape[1]
    n_distribution = n_samples - time_series_len
    if (n_distribution < 1):
        sys.exit("time_series_len =", time_series_len, "cannot exceed input trace length =", n_samples)
    random_lag = np.random.random_integers(-max_shift, +max_shift, size=n_obs*n_duplicate)
    X_out = np.zeros([len(random_lag), time_series_len], dtype='float')
    ibeg = int(n_samples/2) - int(time_series_len/2) # e.g., 100
    print("Beginning sample to which random lags are added:", ibeg)
    print("Min/max lag:", min(random_lag), max(random_lag))
    for iduplicate in range(n_duplicate):
        for iobs in range(n_obs):
            isrc = iobs
            idst = iduplicate*n_obs + iobs 
            # In some respect, the sign doesn't matter.  But in practice, it will
            # conceptually simpler if we add a correction to an initial pick.
            # If the lag is -0.3 (-30 samples), ibeg = 100, and t_pick = 200, then
            # the trace will start at 100 - -30 = 130 samples.  The pick will be at
            # 200 - 130 = 70 samples instead of 100 samples.  In this case, let's 
            # assume the pick, t_0, is very late.  The new pick will be corrected
            # by adding the result of the network to the pick - i.e,. t_0 + (-0.3).
            i1 = ibeg - random_lag[idst] # shift is t - tau
            i2 = i1 + time_series_len
            X_out[idst,:] = X_in[iobs, i1:i2]
            # Remember to normalize
            xnorm = np.max(np.abs(X_out[idst,:]))
            X_out[idst,:] = X_out[idst,:]/xnorm
    return X_out, random_lag*dt

def loaders(
    train_file,
    validation_file,
    path,
    batch_size,
    num_workers,
    time_series_len,
    max_dt,
    dt, 
    n_duplicate_train, 
    shuffle_train=True
):

    np.random.seed(2482045)
    train_file = h5py.File(f'{path}/{train_file}', 'r')
    print('Train shape:', train_file['X'].shape)
    X_waves_train = train_file['X'][:]#[0:80000]
    # Y_train = train_file['Y'][:]#[0:80000]
    train_file.close()
    print("Randomizing start times...")
    X_train, Y_train = randomize_start_times_and_normalize(X_waves_train,
                                                           time_series_len = time_series_len,
                                                           max_dt = max_dt, dt = dt,
                                                           n_duplicate = n_duplicate_train)
    train_dataset = NumpyDataset(X_train, Y_train)


    validation_file = h5py.File('data/uuss_validation.h5', 'r')
    print('Validation shape:', validation_file['X'].shape)
    X_waves_validate = validation_file['X'][:]#[0:3200]
    # Y_validate = validation_file['Y'][:]#[0:3200]
    validation_file.close()
    print("Randomizing start times...")
    X_validate, Y_validate = randomize_start_times_and_normalize(X_waves_validate,
                                                                 time_series_len = time_series_len,
                                                                 max_dt = max_dt, dt = dt,
                                                                 n_duplicate = 1)
    print("Creating validation dataset...")
    validation_dataset = NumpyDataset(X_validate, Y_validate)

    return {
        "train": torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )}
        