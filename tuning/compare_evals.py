#%%
import glob
import numpy as np
import sys
sys.path.append("../swag_modified/swag")
import utils


for  file in glob.glob("*/val_eval.npz"):
    print(file)
    file = np.load(file)
    pred_mean = file["prediction_mean"]
    targets = file["targets"]
    resids = targets[:, 0] - pred_mean
    of_mean, of_std = utils.compute_outer_fence_mean_standard_deviation(resids)
    print("Stats for sgd ensemble residuals")
    print("Mean    STD    OF_Mean    OF_STD")
    print(np.mean(resids), np.std(resids), of_mean, of_std)

#%%
