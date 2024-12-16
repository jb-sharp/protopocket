"""
Analyze GENEOmap results on ProSPECCTs benchmarck.
@author: Giovanni Bocchi
@institution: University of Milan
@email: giovanni.bocchi1@unimi.it
"""

import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import torch
from geneomap_training import GENEOmap

sns.set_theme(context = "talk",
              style = "ticks",
              palette = "colorblind",
              font_scale = 1)

prefix = os.getcwd()

SIZE        = 500
N           = 1000
fpr_mean    = np.linspace(0, 1, N)

paths = sorted(glob.glob(f"{prefix}/models/model-ounit32-ts1-{SIZE}-ts2-{SIZE}-mauc*-s???.dat"))
nmodels = len(paths)
print(f"MODELS: {nmodels}")

datasets = [("identical_structures", "D1", 0 ),
            ("identical_structures_similar_ligands", "D1.2", 1 ),
            ("NMR_structures", "D2",2 ),
            ("decoy_shape_structures", "D3", 3 ),
            ("decoy_rational_structures", "D4", 4 ),
            ("kahraman_structures", "D5", 5 ),
            ("barelier_structures", "D6", 6 ),
            ("review_structures", "D7", 7 ) ]

results = {}

for i, path in enumerate(paths):

    checkpoint = torch.load(path,
                        weights_only = False)

    model = GENEOmap((np.zeros((100, 32)),
                      np.zeros((100, 32)),
                      np.zeros((32,))))
    model.load_state_dict(checkpoint["model_state_dict"])

    aucs = []
    interp_tprs = []

    results[f"model{i+1}"] = {}

    for dataset, label, j in datasets:

        results[f"model{i+1}"][dataset] = {}

        for test in [True, False]:

            results[f"model{i+1}"][dataset][test] = {}

            path1 = f"{prefix}/data/geneomap_{dataset}.csv"
            path2 = f"{prefix}/prospectts/{dataset}.csv"
            df = pd.read_csv(path1, header = 0, index_col = 0)

            scaler = StandardScaler()
            xscaled = scaler.fit_transform(df)
            dfscaled = pd.DataFrame(xscaled, columns = df.columns)
            dfscaled.index = df.index
            truth = pd.read_csv(path2, header = None)
            truth.columns = ["P1", "P2", "label"]

            yn  = (truth.label == "inactive").to_numpy()
            y = yn.astype(np.float32).reshape((-1, 1)).astype(np.float32)
            f1 = dfscaled.loc[truth.P1].to_numpy().astype(np.float32)
            f2 = dfscaled.loc[truth.P2].to_numpy().astype(np.float32)

            if test:
                i0, i1 = checkpoint["test_set"][j]
                ite = np.concatenate([i0, i1])
            else:
                i0 = np.where(y == 0)[0]
                i1 = np.where(y == 1)[0]
                ite = np.arange(len(y))

            s = model.score(torch.as_tensor(f1[ite]), torch.as_tensor(f2[ite]))
            a = model.activation(s).detach().cpu()
            fpr, tpr, thresholds = metrics.roc_curve(y[ite], a, pos_label = 1)
            auc = metrics.auc(fpr, tpr)
            aucs.append(auc)

            interp_tpr    = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)

            results[f"model{i+1}"][dataset][test]["tpr"] = interp_tpr
            results[f"model{i+1}"][dataset][test]["fpr"] = fpr
            results[f"model{i+1}"][dataset][test]["thresholds"] = thresholds
            results[f"model{i+1}"][dataset][test]["auc"] = auc

### PLOTS

TEST = False   # If True plots validations results else whole datasets ones

fig, ax = plt.subplots(3, 3)
auc_means_train = []
auc_vars_train  = []
auc_means_test  = []
auc_vars_test   = []

for dataset, label, i in datasets:

    I = i // 3
    J = i %  3

    aucs_train        = [results[f"model{i+1}"][dataset][False]["auc"] for i in range(nmodels)]
    aucs_test         = [results[f"model{i+1}"][dataset][True]["auc"] for i in range(nmodels)]
    interp_tprs_test  = [results[f"model{i+1}"][dataset][True]["tpr"] for i in range(nmodels)]
    interp_tprs_train = [results[f"model{i+1}"][dataset][False]["tpr"] for i in range(nmodels)]

    auc_mean_train     = np.mean(aucs_train)
    auc_se_train       = np.std(aucs_train) / np.sqrt(nmodels)
    auc_mean_test      = np.mean(aucs_test)
    auc_se_test        = np.std(aucs_test) / np.sqrt(nmodels)

    tpr_mean_test     = np.mean(interp_tprs_test, axis = 0)
    tpr_mean_test[-1] = 1.0
    tpr_std_test      = np.std(interp_tprs_test, axis = 0) / np.sqrt(nmodels)
    tpr_upper_test    = np.clip(tpr_mean_test + 2*tpr_std_test, 0, 1)
    tpr_lower_test    = tpr_mean_test - 2*tpr_std_test

    tpr_mean_train     = np.mean(interp_tprs_train, axis = 0)
    tpr_mean_train[-1] = 1.0
    tpr_std_train      = np.std(interp_tprs_train, axis = 0) / np.sqrt(nmodels)
    tpr_upper_train    = np.clip(tpr_mean_train + 2*tpr_std_train, 0, 1)
    tpr_lower_train    = tpr_mean_train - 2*tpr_std_train

    auc_means_train.append(np.mean(aucs_train))
    auc_vars_train.append(np.var(aucs_train))
    auc_means_test.append(np.mean(aucs_test))
    auc_vars_test.append(np.var(aucs_test))

    if not TEST:
        ax[I, J].plot(fpr_mean, tpr_mean_train, color = f"C{i}",
                        label = label)
        ax[I, J].fill_between(fpr_mean, tpr_lower_train, tpr_upper_train,
                                alpha = 0.2, color = f"C{i}")
        ax[I, J].set_title(fr"AUC {auc_mean_train:.2f} $\pm$ {auc_se_train:.2f}")
    else:
        ax[I, J].plot(fpr_mean, tpr_mean_test, color = f"C{i}",
                        label = label)
        ax[I, J].fill_between(fpr_mean, tpr_lower_test, tpr_upper_test,
                                alpha = 0.2, color = f"C{i}")
        ax[I, J].set_title(fr"AUC {auc_mean_test:.2f} $\pm$ {auc_se_test:.2f}")

    ax[I, J].plot([0, 1], [0, 1], "k--")
    ax[I, J].legend(loc = "lower right")

ax[2, 2].remove()

## Average AUC Mean and SE for whole datasets
auc_mean_train = np.mean(auc_means_train)
auc_se_train = np.sqrt(np.sum(np.array(auc_vars_train) / nmodels) / len(datasets)**2)

## Average AUC Mean and SE for validations
auc_mean_test = np.mean(auc_means_test)
auc_se_test = np.sqrt(np.sum(np.array(auc_vars_test) / nmodels) / len(datasets)**2)
