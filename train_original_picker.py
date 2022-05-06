#!/usr/bin/env python3
import numpy as np
import warnings
import sys
import os
import time
import h5py
import torch
import torch.utils.data
import sklearn as sk
from sklearn.metrics import confusion_matrix

warnings.simplefilter("ignore")

def get_n_params(model):
    """
    Computes the number of trainable model parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_outer_fence_mean_standard_deviation(residuals):
    """ 
    Computes the mean and standard deviation using the outer fence method.
    The outerfence is [25'th percentile - 1.5*IQR, 75'th percentile + 1.5*IQR]
    where IQR is the interquartile range.

    Parameters
    ----------
    residuals : The travel time residuals in seconds.

    Results
    -------
    mean : The mean (seconds) of the residuals in the outer fence.
    std : The standard deviation (seconds) of the residuals in the outer fence.  
    """
    q1, q3 = np.percentile(residuals, [25,75])
    iqr = q3 - q1
    of1 = q1 - 1.5*iqr
    of3 = q3 + 1.5*iqr
    trimmed_residuals = residuals[(residuals > of1) & (residuals < of3)]
    #print(len(trimmed_residuals), len(residuals), of1, of3)
    mean = np.mean(trimmed_residuals)
    std = np.std(trimmed_residuals)
    return mean, std

def compute_snr(X_in, y_pick, t_noise = 0.4, t_signal = 0.5,
                dt = 0.01, t_shift_pick = 0.2, min_snr =-20, max_snr = 50):
    """
    Computes the SNR for a signal.

    Parameters
    ----------
    X_in : The matrix of signals.
    y_pick : The pick time of relative to the trace start in seconds.
    t_noise : The noise window for computing the noise energy. 
    t_signal : The signal window for computing the signal energy.
    dt : The sampling period in seconds.
    t_shift_pick : This will set the pick time to to y_pick - t_shift_pick time.
                   This is a safety margin.
    min_snr : Minimum SNR allowed.  Some of these are bonkers.
    max_snr : Maximum SNR allowed.  Some of these are bonkers.
    
    Results
    -------
    snr : The signal to noise ratio for each signal.
    """

    n_waves = X_in.shape[0]
    n_samples = X_in.shape[1]
    n_noise = int(t_noise/dt) + 1 
    n_signal = int(t_signal/dt) + 1
    snr = np.zeros(n_waves)
    for i in range(n_waves):
        pick = int((y_pick[i] - t_shift_pick)/dt)
        i0 = max(0, pick - n_noise)
        i1 = pick
        j0 = pick
        j1 = min(n_samples, pick + n_signal)
        noise  = X_in[i, i0:i1] - np.mean(X_in[i, i0:i1])
        signal = X_in[i, j0:j1] - np.mean(X_in[i, j0:j1])
        n2 = np.multiply(noise, noise)/len(noise)
        s2 = np.multiply(signal, signal)/len(signal)
        en  = np.mean(n2)
        es = np.mean(s2)
        snr[i] = 10*np.log10(es/en)
        if (snr[i] < min_snr):
            snr[i] = min_snr
        if (snr[i] > max_snr):
            snr[i] = max_snr
    return snr 
        
     
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

class CNNNet(torch.nn.Module):

    def __init__(self, num_channels=1, min_lag = -0.5, max_lag = 0.5):
        super(CNNNet, self).__init__()
        from torch.nn import MaxPool1d, Conv1d, Linear
        self.relu = torch.nn.ReLU()
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.Hardtanh = torch.nn.Hardtanh(min_val = self.min_lag, max_val = self.max_lag)
        filter1 = 21
        filter2 = 15
        filter3 = 11

        self.maxpool = MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = Conv1d(num_channels, 32,
                            kernel_size=filter1, padding=filter1//2)
        self.bn1 = torch.nn.BatchNorm1d(32, eps=1e-05, momentum=0.1)
        # Output has dimension [200 x 32]

        
        self.conv2 = Conv1d(32, 64,
                            kernel_size=filter2, padding=filter2//2)
        self.bn2 = torch.nn.BatchNorm1d(64, eps=1e-05, momentum=0.1)
        # Output has dimension [100 x 64] 

        self.conv3 = Conv1d(64, 128,
                            kernel_size=filter3, padding=filter3//2)
        self.bn3 = torch.nn.BatchNorm1d(128, eps=1e-05, momentum=0.1)
        # Output has dimension [50 x 128]

        self.fcn1 = Linear(6400, 512)
        self.bn4 = torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1)
  
        self.fcn2 = Linear(512, 512)
        self.bn5 = torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1)

        self.fcn3 = Linear(512, 1)

    def forward(self, x):
        # N.B. Consensus seems to be growing that BN goes after nonlinearity
        # That's why this is different than Zach's original paper.
        # First convolutional layer
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        # Second convolutional layer
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpool(x)
        # Third convolutional layer
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.maxpool(x)
        # Flatten
        x = x.flatten(1) #torch.nn.flatten(x)
        # First fully connected layer
        x = self.fcn1(x)
        x = self.relu(x)
        x = self.bn4(x)
        # Second fully connected layer
        x = self.fcn2(x)
        x = self.relu(x)
        x = self.bn5(x)
        # Last layer
        x = self.fcn3(x)
        # Force linear layer to be between +/- 0.5
        x = self.Hardtanh(x)
        return x

    def freeze_convolutional_layers(self):
        self.conv1.weight.requires_grad = False
        self.conv1.bias.requires_grad = False
        self.bn1.weight.requires_grad = False
        self.bn1.bias.requires_grad = False
        # Second convolutional layer
        self.conv2.weight.requires_grad = False
        self.conv2.bias.requires_grad = False
        self.bn2.weight.requires_grad = False
        self.bn2.bias.requires_grad = False
        # Third convolutional layer
        self.conv3.weight.requires_grad = False
        self.conv3.bias.requires_grad = False
        self.bn3.weight.requires_grad = False
        self.bn3.bias.requires_grad = False

    def write_weights_to_hdf5(self, file_name, bias):
        f = h5py.File(file_name, 'w')
 
        g1 = f.create_group("/model_weights")
        g = g1.create_group("sequential_1")

        g.create_dataset("conv1d_1.weight", data=np.array(self.conv1.weight.data.cpu()))
        g.create_dataset("conv1d_1.bias", data=np.array(self.conv1.bias.data.cpu()))
        g.create_dataset("bn_1.weight", data=np.array(self.bn1.weight.data.cpu())) # gamma
        g.create_dataset("bn_1.bias", data=np.array(self.bn1.bias.data.cpu()))  # beta
        g.create_dataset("bn_1.running_mean", data=np.array(self.bn1.running_mean.data.cpu()))
        g.create_dataset("bn_1.running_var", data=np.array(self.bn1.running_var.data.cpu()))

        g.create_dataset("conv1d_2.weight", data=np.array(self.conv2.weight.data.cpu()))
        g.create_dataset("conv1d_2.bias", data=np.array(self.conv2.bias.data.cpu()))
        g.create_dataset("bn_2.weight", data=np.array(self.bn2.weight.data.cpu())) # gamma
        g.create_dataset("bn_2.bias", data=np.array(self.bn2.bias.data.cpu()))  # beta
        g.create_dataset("bn_2.running_mean", data=np.array(self.bn2.running_mean.data.cpu()))
        g.create_dataset("bn_2.running_var", data=np.array(self.bn2.running_var.data.cpu()))

        g.create_dataset("conv1d_3.weight", data=np.array(self.conv3.weight.data.cpu()))
        g.create_dataset("conv1d_3.bias", data=np.array(self.conv3.bias.data.cpu()))
        g.create_dataset("bn_3.weight", data=np.array(self.bn3.weight.data.cpu())) # gamma
        g.create_dataset("bn_3.bias", data=np.array(self.bn3.bias.data.cpu()))  # beta
        g.create_dataset("bn_3.running_mean", data=np.array(self.bn3.running_mean.data.cpu()))
        g.create_dataset("bn_3.running_var", data=np.array(self.bn3.running_var.data.cpu()))

        g.create_dataset("fcn_1.weight", data=np.array(self.fcn1.weight.data.cpu()))
        g.create_dataset("fcn_1.bias", data=np.array(self.fcn1.bias.data.cpu()))
        g.create_dataset("bn_4.weight", data=np.array(self.bn4.weight.data.cpu())) # gamma
        g.create_dataset("bn_4.bias", data=np.array(self.bn4.bias.data.cpu()))  # beta
        g.create_dataset("bn_4.running_mean", data=np.array(self.bn4.running_mean.data.cpu()))
        g.create_dataset("bn_4.running_var", data=np.array(self.bn4.running_var.data.cpu()))

        g.create_dataset("fcn_2.weight", data=np.array(self.fcn2.weight.data.cpu()))
        g.create_dataset("fcn_2.bias", data=np.array(self.fcn2.bias.data.cpu()))
        g.create_dataset("bn_5.weight", data=np.array(self.bn5.weight.data.cpu())) # gamma
        g.create_dataset("bn_5.bias", data=np.array(self.bn5.bias.data.cpu()))  # beta
        g.create_dataset("bn_5.running_mean", data=np.array(self.bn5.running_mean.data.cpu()))
        g.create_dataset("bn_5.running_var", data=np.array(self.bn5.running_var.data.cpu()))

        g.create_dataset("fcn_3.weight", data=np.array(self.fcn3.weight.data.cpu()))
        g.create_dataset("fcn_3.bias", data=np.array(self.fcn3.bias.data.cpu()))

        g2 = f.create_group("/model_bias")
        g2.create_dataset("bias", data=bias)

        f.close()

class Model():
    def __init__(self, network, optimizer, model_path, device):
        self.network = network
        self.optimizer = optimizer
        self.model_path = model_path
        self.device = device

    def train(self, train_loader, val_loader, n_epochs):
        from torch.autograd import Variable

        self.network.train()
        print("Number of trainable parameters:", get_n_params(self.network))
        # Want to avoid L2 particularly when we go to the archives which have
        # some, err, interesting examples
        loss = torch.nn.MSELoss() # Want to avoid L2 particularly when we go to the past
        # When |r| > delta switch from L2 to L1 norm.  Typically, autopicks will
        # be within 0.1 seconds of analyst picks so this seems generous but also
        # will also downweight any really discordant examples
        #loss = torch.nn.HuberLoss(delta = 0.25)
        n_batches = len(train_loader)
        training_start_time = time.time()
        training_rms_baseline = None
        validation_rms_baseline = None

        print(self.network.min_lag, self.network.max_lag)

        print_every = n_batches//10
        if (not os.path.exists(self.model_path)):
            os.makedirs(self.model_path)
       

        for epoch in range(n_epochs):
            print("Beginning epoch {}...".format(epoch+1))
            running_accuracy = 0
            running_loss = 0
            running_sample_count = 0
            total_training_loss = 0
            total_validation_loss = 0
            start_time = time.time()

            n_total_pred = 0
            y_true_all = np.zeros(len(train_loader.dataset), dtype='float')
            y_est_all = np.zeros(len(train_loader.dataset), dtype='float')
            for i, data in enumerate(train_loader, 0):
                # Get inputs/outputs and wrap in variable object
                inputs, y_true = data
                inputs, y_true = Variable(
                    inputs.to(self.device)), Variable(
                    y_true.to(self.device))

                # Set gradients for all parameters to zero
                #print(inputs.shape)
                if (inputs.shape[0] < 2):
                    print("Skipping edge case")
                    continue
                self.optimizer.zero_grad()

                # Forward pass
                y_est = self.network(inputs)

                # Backward pass
                loss_value = loss(y_est, y_true)
                loss_value.backward()
                loss_scalar_value = loss_value.data.cpu().numpy()

                # Update parameters
                self.optimizer.step()

                # Print statistics
                with torch.no_grad():
                    running_loss += loss_scalar_value
                    total_training_loss += loss_scalar_value

                    running_accuracy += loss_scalar_value
                    running_sample_count += torch.numel(y_true)
                    for i_local_pred in range(len(y_true)):
                        y_true_all[n_total_pred] = y_true[i_local_pred].cpu().numpy()
                        y_est_all[n_total_pred]  = y_est[i_local_pred].cpu().numpy()
                        #print(y_true_all[n_total_pred], y_est_all[n_total_pred])
                        n_total_pred = n_total_pred + 1

                # Print every n'th batch of an epoch
                if ( (i + 1)%(print_every + 1) == 0):
                    print("Epoch {}, {:d}% \t train_loss: {:.4e} "
                          "train_error: {:4.2f} took: {:.4f}s".format(
                          epoch + 1, int(100*(i + 1)/n_batches),
                          running_loss/print_every,
                          100*running_accuracy/running_sample_count,
                          time.time() - start_time) )
                    running_loss = 0
                    start_time = time.time()
                # end print epoch
            # Loop on no gradient
            # Resize 
            y_true_all = y_true_all[0:n_total_pred]
            y_est_all = y_est_all[0:n_total_pred]
            residuals = y_true_all - y_est_all
            training_mean = np.mean(residuals)
            training_std = np.std(residuals)
            training_mean_of, training_std_of = compute_outer_fence_mean_standard_deviation(residuals) 
            training_rms = np.sqrt( np.sum(residuals**2)/len(y_true_all) )
            random_lags = np.random.random_integers(self.network.min_lag, self.network.max_lag, size=n_total_pred)
            if (training_rms_baseline is None):
                residuals = y_true_all - random_lags 
                training_mean_baseline = np.mean(residuals)
                training_std_baseline = np.std(residuals)
                training_rms_baseline = np.sqrt( np.sum(residuals**2)/len(y_true_all) )
            print("Training for epoch (Mean,Std,Outer Fence Mean, Outer Fence Std,RMS,Loss): (%f,%f,%f,%f,%f,%f) (Baseline Mean,Std,RMS~ %f,%f,%f)"%(
                  training_mean, training_std, training_mean_of, training_std_of,
                  training_rms, total_training_loss, 
                  training_mean_baseline, training_std_baseline, training_rms_baseline))

            # Validation
            running_sample_count = 0
            running_val_accuracy = 0
            n_total_pred = 0
            y_true_all = np.zeros(len(val_loader.dataset), dtype='float')
            y_est_all = np.zeros(len(val_loader.dataset), dtype='float')
            with torch.no_grad():
                for inputs, y_true in val_loader:

                    # Wrap tensors in Variables
                    inputs, y_true = Variable(
                        inputs.to(device)), Variable(
                        y_true.to(device))

                    # Forward pass only
                    y_est = self.network(inputs)
                    val_loss = loss(y_est, y_true)
                    total_validation_loss += val_loss.item()

                    for i_local_pred in range(len(y_true)):
                        y_true_all[n_total_pred] = y_true[i_local_pred].cpu().numpy()
                        y_est_all[n_total_pred]  = y_est[i_local_pred].cpu().numpy()
                        n_total_pred = n_total_pred + 1
            # Loop on data in training
            y_true_all = y_true_all[0:n_total_pred]
            y_est_all = y_est_all[0:n_total_pred]
            residuals = y_true_all - y_est_all # Add mean
            validation_mean = np.mean(residuals)
            validation_std = np.std(residuals)
            validation_mean_of, validation_std_of = compute_outer_fence_mean_standard_deviation(residuals)
            validation_rms = np.sqrt(np.sum(residuals**2)/len(y_true_all))
            random_lags = np.random.random_integers(self.network.min_lag, self.network.max_lag, size=n_total_pred)
            if (validation_rms_baseline is None):
                residuals = y_true_all - random_lags
                validation_mean_baseline = np.mean(residuals)
                validation_std_baseline = np.std(residuals)
                validation_rms_baseline = np.sqrt( np.sum(residuals**2)/len(y_true_all) )
            print("Validation (Mean,Std,Outer Fence Mean, Outer Fence Std,RMS,Loss): (%f,%f,%f,%f,%f,%f) (Baseline Mean,Std,RMS ~ %f,%f,%f)"%(
                  validation_mean, validation_std, validation_mean_of, validation_std_of,
                  validation_rms, total_validation_loss,
                  validation_mean_baseline, validation_std_baseline, validation_rms_baseline))

            model_file_name = os.path.join(self.model_path,
                                           'models_%03d.pt'%(epoch+1))
            torch.save({
                       'epoch': epoch+1,
                       'model_state_dict': self.network.state_dict(),
                       'optimizer_state_dict': self.optimizer.state_dict(),
                       'training_mean_of': training_mean_of,
                       'validation_mean_of': validation_mean_of,
                       'training_std_of': training_std_of,
                       'validation_std_of': validation_std_of,
                       'training_mean': training_mean,
                       'validation_mean': validation_mean,
                       'training_std': training_std,
                       'validation_std': validation_std,
                       'validation_loss': total_validation_loss,
                       'training_loss': total_training_loss,
                       'validation_rms': validation_rms,
                       'training_rms': training_rms,
                       }, model_file_name)
            # Write the validation_mean_of as the bias to add to picks during
            # application since, if I did my job right, the validation dataset
            # looks a lot like the test dataset and real datasets we'll see in
            # the wild.  Additionally, while the training mean can correspond
            # to the validation mean they tend to diverge when the model is
            # under or overfit.
            self.network.write_weights_to_hdf5(os.path.join(self.model_path, 
                                                            'models_%03d.h5'%(epoch+1)),
                                                            validation_mean_of)
        # Loop on epochs

if __name__ == "__main__":
    #device = torch.device("cpu") #torch.device("cuda:1")
    device = torch.device("cuda:0")
    np.random.seed(82323) 
    time_series_len = 400
    n_epochs = 15
    n_duplicate_train = 2 # Number of times to duplicate (but augment) the training dataset
    max_dt = 0.5 # Allow +/- start time perturbation in data augmentation
    max_dt_nn = 0.75 # The network must make predictions within this bound 
    dt = 0.01 # Sampling period
    fine_tune = True
    freeze_convolutional_layers = False # Freeze convolutional layers during fine tuning?
    model_to_fine_tune = 'caModels/models_004.pt' #'models/models_011.pt'

    if (not fine_tune):
        print("Loading California training data...")
        learning_rate = 0.000075
        n_duplicate_train = 2 # Seems to be all my GPU can handle
        train_file = h5py.File('data/scsn_p_2000_2017_6sec_0.5r_pick_train.hdf5', 'r')
    else:
        np.random.seed(2482045)
        print("Loading Utah training data...")
        train_file = h5py.File('data/uuss_train.h5', 'r')
        learning_rate = 0.00002
        n_duplicate_train = 3

    print('Train shape:', train_file['X'].shape)
    X_waves_train = train_file['X'][:]#[0:80000]
    Y_train = train_file['Y'][:]#[0:80000]
    train_file.close()
    print("Randomizing start times...")
    X_train, Y_train = randomize_start_times_and_normalize(X_waves_train,
                                                           time_series_len = time_series_len,
                                                           max_dt = max_dt, dt = dt,
                                                           n_duplicate = n_duplicate_train)
    print("Creating training dataset...") 
    train_dataset = NumpyDataset(X_train, Y_train)
    indices = list(range(len(Y_train)))
    train_index = np.random.choice(indices, size=len(Y_train), replace=False)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_index)
    print(train_sampler)
    params_train = {'batch_size': 64,
                    'shuffle': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, **params_train)

    if (not fine_tune):
        print("Loading CA validation data...")
        validation_file = h5py.File('data/scsn_p_2000_2017_6sec_0.5r_pick_test.hdf5', 'r')
    else:
        validation_file = h5py.File('data/uuss_validation.h5', 'r')
    print('Validation shape:', validation_file['X'].shape)
    X_waves_validate = validation_file['X'][:]#[0:3200]
    Y_validate = validation_file['Y'][:]#[0:3200]
    validation_file.close()
    print("Randomizing start times...")
    X_validate, Y_validate = randomize_start_times_and_normalize(X_waves_validate,
                                                                 time_series_len = time_series_len,
                                                                 max_dt = max_dt, dt = dt,
                                                                 n_duplicate = 1)
    print("Creating validation dataset...")
    validation_dataset = NumpyDataset(X_validate, Y_validate)
    indices = list(range(len(Y_validate))) 
    validation_index = np.random.choice(indices, size=len(Y_validate), replace=False)
    validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(validation_index)
    params_validate = {'batch_size': 512,
                       'shuffle': True}
    validation_loader = torch.utils.data.DataLoader(validation_dataset, **params_validate)

    #print(X_validate.shape)
    #print(Y_validate[0:10])
    #print(np.sum(Y_validate == 0)) # Up
    #print(np.sum(Y_validate == 1)) # Down
    #print(np.sum(Y_validate == 2)) # Unknown
    #stop
    # Create the network and optimizer
    cnnnet = CNNNet(min_lag = -max_dt_nn, max_lag = +max_dt_nn).to(device)
    print("Number of model parameters:", get_n_params(cnnnet))
    optimizer = torch.optim.Adam(cnnnet.parameters(), lr=learning_rate)
    model_path = './caModels'
    if (fine_tune):
        print("Will fine tune:", model_to_fine_tune)
        if (not model_to_fine_tune is None):
            print("Loading model: ", model_to_fine_tune)
            check_point = torch.load(model_to_fine_tune)
            cnnnet.load_state_dict(check_point['model_state_dict'])
        model_path = './uussModels'
        print("Will write models to:", model_path)
        if (freeze_convolutional_layers):
            print("Freezing convolutional layers...")
            cnnnet.freeze_convolutional_layers()
    model = Model(cnnnet, optimizer, model_path=model_path, device=device)
    print("Number of trainable parameters:", get_n_params(cnnnet))
    print("Starting training...")
    model.train(train_loader, validation_loader, n_epochs)

    #i1 = 0
    #i2 = 32
    #X = np.zeros([32, 1, 400])
    #y = np.zeros([32, 1])
    #X = torch.from_numpy(X).float().to(device)
    #res = cnnnet.forward(X)
    #print(res.argmax(1))
    #print(res)
