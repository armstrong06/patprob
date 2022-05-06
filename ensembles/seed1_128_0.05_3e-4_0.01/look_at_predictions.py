
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 
import os
import sys
sys.path.append("../../swag_modified/swag")
from utils import compute_outer_fence_mean_standard_deviation
figdir="/home/armstrong/Research/git_repos/patprob/ensembles/seed1_128_0.05_3e-4_0.01/figures"
if not os.path.exists(figdir):
    os.makedirs(figdir)
#%%   
file = np.load("swag_test_uncertainty.npz")
predictions = file["predictions"]
targets = file["targets"][:, 0]
pred_mean = file["prediction_mean"]
pred_std = file["prediction_std"]

print(predictions.shape)

resids = file["targets"][:, 0] - file["prediction_mean"]

# %%

plt.hist(resids, density=True, bins=np.arange(-0.95, 0.96, 0.01));
plt.ylabel("Density", fontsize=14)
plt.xlabel("Seconds", fontsize=14)
#%%
plt.hist(resids, density=True, bins=np.arange(-0.5, 0.51, 0.01), edgecolor="k" );
plt.ylabel("Density", fontsize=14)
plt.xlabel("Seconds", fontsize=14)
#%%
print(np.mean(resids))
print(np.std(resids))
print(compute_outer_fence_mean_standard_deviation(resids))
#%%
import h5py
f = h5py.File("../../data/uuss_test_fewerhist.h5", "r")
X = f["X"][:]
f.close()
# %%
wf_len = X.shape[1]
wf_center = wf_len//2
#%%

for i in range(1000):
    shift = targets[i]
    shifted_predictions = predictions[i, :]-shift
    shifted_pick = pred_mean[i]-shift
    std = pred_std[i]

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
    ax.plot(hist_range, norm.pdf(hist_range, pred_mean[i]-shift, pred_std[i]), color="r")
    ax.text(0.05, 0.9, f"std={str(round(std, 3))}", transform=ax.transAxes, fontsize=12)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_xlabel("Seconds", fontsize=14)
    plt.savefig(f"{figdir}/test_wf_{i}.jpg")
    plt.close()
# %%

import pandas as pd

meta_df = pd.read_csv("../../data/uuss_test_fewerhist.csv")

#%%

plt.scatter(meta_df.pick_quality, pred_std, alpha=0.1)
# %%
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
def density_scatter( x , y, ax = None, sort = True, bins = 20, color_bar=False,**kwargs )   :
    """
    Scatter plot colored by 2d histogram
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
ax = density_scatter(meta_df[meta_df["pick_quality"] == 1]["pick_quality"].values, pred_std[one_inds], color_bar=True)
print(len(one_inds))

one_inds = meta_df[meta_df["pick_quality"] == 0.75].index
density_scatter(meta_df[meta_df["pick_quality"] == 0.75]["pick_quality"].values, pred_std[one_inds], ax=ax)
print(len(one_inds))

one_inds = meta_df[meta_df["pick_quality"] == 0.5].index
density_scatter(meta_df[meta_df["pick_quality"] == 0.5]["pick_quality"].values, pred_std[one_inds], ax=ax)
print(len(one_inds))


ax.set_ylabel("Prediction STD", fontsize=14)
ax.set_xlabel("Pick Quality", fontsize=14)

ax.set_xticks([0.5, 0.75, 1.0])
ax.set_ylim([0, 0.04])
# %%
