
#%%
import numpy as np

file = np.load("swag_uncertainty.npz")

#%%

file["predictions"]
# %%
len(file["prediction_mean"])
# %%
targets = file["targets"][:, 0]
preds = file["prediction_mean"]
pred_std = file["prediction_std"]
# %%
resids = file["targets"][:, 0] - file["prediction_mean"]
# %%
print(resids)
# %%
import matplotlib.pyplot as plt
from scipy.stats import norm 
plt.hist(resids, density=True, bins=50)
# %%
for i in range(5):
    bins = plt.hist(file["predictions"][i, :], bins=30, density=True)
    plt.axvline(preds[i], color="red")
    plt.axvline(targets[i], color="green")
    plt.plot(bins[1], norm.pdf(bins[1], preds[i], pred_std[i]))
    plt.show()
# %%
