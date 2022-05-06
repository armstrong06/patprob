
#%%
from math import comb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 
import os

import sys
sys.path.append("../swag_modified/swag")
from utils import compute_outer_fence_mean_standard_deviation


figdir="/home/armstrong/Research/git_repos/patprob/ensembles/figures"
if not os.path.exists(figdir):
    os.makedirs(figdir)
#%%  
def load_model_results(file_name):
    file = np.load(file_name)
    predictions = file["predictions"]
    targets = file["targets"][:, 0]
    pred_mean = file["prediction_mean"]
    pred_std = file["prediction_std"]
    resids = targets - pred_mean

    return predictions, pred_mean, pred_std, targets, resids

model1_file = "seed1_128_0.05_3e-4_0.01/swag_test_uncertainty.npz"
model2_file = "seed2_128_0.05_3e-4_0.01/swag_test_uncertainty.npz"
model3_file = "seed3_128_0.05_3e-4_0.01/swag_test_uncertainty.npz"

model1_preds, model1_pred_mean, model1_pred_std, model1_targets, model1_resids = load_model_results(model1_file)
model2_preds, model2_pred_mean, model2_pred_std, model2_targets, model2_resids = load_model_results(model2_file)
model3_preds, model3_pred_mean, model3_pred_std, model3_targets, model3_resids = load_model_results(model3_file)

# %%

combined_predictions = np.concatenate([model1_preds, model2_preds, model3_preds], axis=1)
combined_means = np.mean(combined_predictions, axis=1)
combined_stds = np.std(combined_predictions, axis=1)
assert np.array_equal(model1_targets, model2_targets)
assert np.array_equal(model1_targets, model3_targets)
combined_residuals = model1_targets - combined_means
#%%
plt.hist(combined_residuals, density=True, bins=np.arange(-0.95, 0.96, 0.01));
plt.ylabel("Density", fontsize=14)
plt.xlabel("Seconds", fontsize=14)
#%%
plt.hist(combined_residuals, density=True, bins=np.arange(-0.5, 0.51, 0.01), edgecolor="k" );
plt.ylabel("Density", fontsize=14)
plt.xlabel("Seconds", fontsize=14)
#%%
print(np.mean(combined_residuals))
print(np.std(combined_residuals))
print(compute_outer_fence_mean_standard_deviation(combined_residuals))
#%%
import h5py
f = h5py.File("../data/uuss_test_fewerhist.h5", "r")
X = f["X"][:]
f.close()
# %%
wf_len = X.shape[1]
wf_center = wf_len//2
#%%

for i in range(1000):
    shift = model1_targets[i]
    shifted_predictions = combined_predictions[i, :]-shift
    shifted_pick = combined_means[i]-shift
    std = combined_stds[i]

    fig, ax = plt.subplots(1)

    # horizontal line at 0
    ax.axhline(0, alpha=0.5, color="k")

    # Prediction histogram
    bins = ax.hist(shifted_predictions, bins=30, density=True, alpha=0.7)

    # Trim and scale waveform
    max_dens = np.max(bins[0])
    width = round(np.max(abs(shifted_predictions)) + 0.1, 2)
    wf_width = round(width*100)
    wf = (X[i, wf_center-wf_width:wf_center+wf_width+1])
    wf_norm = max_dens/np.max(abs(wf))

    # Plot waveform
    x_vals = np.arange(-width, round(width+0.01, 2), 0.01)
    ax.plot(x_vals[:len(wf)], wf*wf_norm, color="k")

    # Plot picks
    ax.axvline(shifted_pick, color="red")
    ax.axvline(0, color="green")

    hist_range = np.zeros(len(bins[1])+2)
    hist_range[0] = bins[1][0] - 0.01
    hist_range[-1] = bins[1][-1] + 0.01
    hist_range[1:-1] = bins[1]
    # Plot gaussion over predictions
    ax.plot(hist_range, norm.pdf(hist_range, shifted_pick, std), color="r")
    ax.text(0.05, 0.9, f"std={str(round(std, 3))}", transform=ax.transAxes, fontsize=12)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_xlabel("Seconds", fontsize=14)
    plt.savefig(f"{figdir}/test_wf_{i}.jpg")
    plt.close()
    #plt.show()
# %%

import pandas as pd

meta_df = pd.read_csv("../data/uuss_test_fewerhist.csv")

#%%

plt.scatter(meta_df.pick_quality, combined_stds, alpha=0.1)
# %%
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
def density_scatter( x , y, ax = None, sort = True, bins = 20, color_bar=False,**kwargs )   :
    """
    Scatter plot colored by 2d histogram - modified from 
    https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
    """
    if ax is None :
        fig , ax = plt.subplots(1)
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs )

    ax.scatter(x[0], y[-1], marker="x", color="k", label="Densest value")
    print("densest val", y[-1])

    mean = np.mean(y)
    print(mean)
    ax.scatter(x[0], mean, marker="x", color="r", label="Mean")
    # mean, std = compute_outer_fence_mean_standard_deviation(y)
    # print(mean)
    # ax.plot(x[0], mean, marker="x", color="k")

    if color_bar:
       # norm = Normalize(vmin = np.min(z), vmax = np.max(z))
        norm = Normalize(vmin = 0, vmax = 1)
        cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
        cbar.ax.set_ylabel('Normalzied Density', fontsize=14)
        
        ax.legend(loc=(0.08,0.8))

    return ax
#%%

#fig, ax = plt.subplots(1)
one_inds = meta_df[meta_df["pick_quality"] == 1].index
ax = density_scatter(meta_df[meta_df["pick_quality"] == 1]["pick_quality"].values, combined_stds[one_inds], color_bar=True)
print(len(one_inds))

two_inds = meta_df[meta_df["pick_quality"] == 0.75].index
density_scatter(meta_df[meta_df["pick_quality"] == 0.75]["pick_quality"].values, combined_stds[two_inds], ax=ax)
print(len(two_inds))

three_inds = meta_df[meta_df["pick_quality"] == 0.5].index
density_scatter(meta_df[meta_df["pick_quality"] == 0.5]["pick_quality"].values, combined_stds[three_inds], ax=ax)
print(len(three_inds))


ax.set_ylabel("Prediction STD", fontsize=14)
ax.set_xlabel("Pick Quality", fontsize=14)

ax.set_xticks([0.5, 0.75, 1.0])
ax.set_ylim([0, 0.04])

# %%
fig, axes = plt.subplots(3, 1)
bins = axes[0].hist(combined_stds[one_inds], density=True, bins=100, alpha=0.7);
bins2 = axes[1].hist(combined_stds[two_inds], density=True, bins=bins[1], alpha=0.5);
bins3 = axes[2].hist(combined_stds[three_inds], density=True, bins=bins[1], alpha=0.5);
#plt.xlim([0, 0.1])
axes[0].set_xticks([])
axes[1].set_xticks([])
for ax in axes:
    ax.set_xlim([0, 0.10])
mean_ones = np.mean(combined_stds[one_inds])
mean_twos = np.mean(combined_stds[two_inds])
mean_threes = np.mean(combined_stds[three_inds])
axes[0].axvline(mean_ones)
axes[1].axvline(mean_twos)
axes[2].axvline(mean_threes, label="mean")

bin_halfwidth = (bins[1][1] - bins[1][0])/2
axes[0].axvline(bins[1][np.argmax(bins[0])]+bin_halfwidth, color="red")
axes[1].axvline(bins2[1][np.argmax(bins2[0])]+bin_halfwidth, color="red")
axes[2].axvline(bins3[1][np.argmax(bins3[0])]+bin_halfwidth, color="red", label="densest bin")
axes[2].legend(loc=5)
axes[1].set_ylabel("Density", fontsize=14)
axes[2].set_xlabel("Prediction STD", fontsize=14)

axes[0].text(0.4, 0.75, "quality=1", transform=axes[0].transAxes, fontsize=12)
axes[1].text(0.4, 0.75, "quality=0.75", transform=axes[1].transAxes, fontsize=12)
axes[2].text(0.4, 0.75, "quality=0.5", transform=axes[2].transAxes, fontsize=12)

# %%
std_bins = np.histogram(combined_stds, bins=20)

bin_mean_resid = []
bin_mean_std = []
bin_mean_diff = []
for i in range(20):
    bin_inds = np.where((combined_stds > bins[1][i]) & (combined_stds < bins[1][i+1]))
    bin_mean_std.append(np.mean(combined_stds[bin_inds]))
    bin_mean_resid.append(np.mean(combined_means[bin_inds]))
    bin_mean_diff.append(np.mean(np.mean(combined_stds[bin_inds]) - np.mean(combined_means[bin_inds])))
plt.plot(bin_mean_std, bin_mean_resid, marker="o")
plt.xlabel("mean STD")
plt.ylabel("Mean residual of bin")
plt.axhline(0, linestyle="--", color="k")

# %%
plt.plot(bin_mean_std, bin_mean_diff, marker="o")
plt.xlabel("mean STD")
plt.ylabel("Mean STD - Mean residual of bin")
plt.axhline(0, linestyle="--", color="k")
# %%
