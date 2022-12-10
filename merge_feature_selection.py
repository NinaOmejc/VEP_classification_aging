import os
import pickle
import numpy as np
import pandas as pd
from src.utils import plot_feature_ranks, plot_feature_ranks_notime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
plt.interactive(False)

### IMPORT STUDY SETTINGS
time_switch = '_notime'
data_version = 'v0' + time_switch
classification_type = 'feature_selection_allsubs' + time_switch
exp_version = 'v0'
balancing = 'smote'
ma_win = 0
classifier_names = ['LDA', 'LR', 'Tree', 'AdaBoost', 'XGB', 'RF', 'SVC_lin']
metric_names = ['mi_total'] # 'rfe_ranking_total', 'sfs_support_total',

if "allsubs" in classification_type and not "age" in classification_type:
    target = "stim"
elif "bysub" in classification_type:
    target = "stim"
elif "allsubsage" in classification_type:
    target = "age"

path_main = os.getcwd()
path_results_in = f"{path_main}{os.sep}results{os.sep}{classification_type}{os.sep}{exp_version}{os.sep}"

general_info_filename = f"{path_results_in}classificaton_results_{classification_type}_dat{data_version}_exp{exp_version}_{balancing}_{target}_ma{ma_win}_general_info.pkl"
with open(general_info_filename, 'rb') as f:
    general_info = pickle.load(f)

feature_names = [general_info["feature_names"][i][0] for i in range(len(general_info["feature_names"]))]
label_names = [general_info["label_names"][i][0] for i in range(len(general_info["label_names"]))]
times = general_info["times"] if time_switch == '' else 0
subID = general_info["subID"]
age_compact = general_info["age_compact"]
nSamples = len(times) if time_switch == '' else 1
nSubj = len(np.unique(subID)) if 'bysub' in classification_type else 1
nClassifiers = len(classifier_names)
nFeatures = len(feature_names)
nClasses = len(label_names)

# set up empty pandas dataframe to collect results
bysub_columns = ['sub', 'age'] if 'bysub' in classification_type else []
df_ranking = pd.DataFrame(columns=['time'] + bysub_columns + ['feature', 'model', 'mi'])
df_ranking['model'] = np.tile(np.repeat(classifier_names, nSamples * nFeatures), nSubj)
df_ranking['feature'] = np.repeat(np.tile(feature_names, nSubj*nClassifiers), nSamples)
df_ranking['time'] = np.tile(times, nFeatures*nClassifiers*nSubj)
df_ranking["model"] = df_ranking["model"].astype("category")
df_ranking["feature"] = df_ranking["feature"].astype("category")

if 'bysub' in classification_type:
    df_ranking['age'] = np.repeat(age_compact, nClassifiers * nFeatures * nSamples)
    df_ranking['sub'] = np.repeat(np.unique(subID), nClassifiers * nFeatures * nSamples)
    df_ranking["sub"] = df_ranking["sub"].astype("category")
    df_ranking["age"] = df_ranking["age"].astype("category")

# IMPORT RESULTS
# clfIdx = 0
# clfName = classifier_names[clfIdx]
for clfIdx, clfName in enumerate(classifier_names):

    filename_results_in = f"classificaton_results_{classification_type}_dat{data_version}_exp{exp_version}_{balancing}_{target}_ma{ma_win}_{clfName}.pkl"
    try:
        with open(f"{path_results_in}{filename_results_in}", 'rb') as f:
            [results, temp1, temp2, temp3, temp4, duration, temp5] = pickle.load(f)
        print(f"Loading results: {clfIdx}, {clfName}")
    except:
        print(f"Loading results: {clfIdx}, {clfName} didnt pass. Seems the file is missing. Skipping.")
        continue

    # melt array into 1D and save for each classifier
    transpose_indices = [1, 0, 2] if 'bysub' in classification_type else [1, 0]
    df_ranking.loc[df_ranking["model"] == clfName, 'mi'] = results[metric_names[0]].transpose(transpose_indices).flatten('F')

# SAVE RESULTS INTO COMMON MERGED PICKLE FILE
with open(f"{path_results_in}merged_feature_ranking_dat{data_version}_exp{exp_version}_{balancing}_{target}_ma{ma_win}.pkl", 'wb') as f:
    pickle.dump([general_info, df_ranking], f, protocol=-1)


metric = "mi"
if time_switch == '_notime':
    plot_feature_ranks_notime(df_ranking, metric, feature_names)
    plot_version = 'v4'
    plt.savefig(f"{path_results_in}plot_feature_selection_{classification_type}_dat{data_version}_exp{exp_version}_{metric}_plot{plot_version}.png",
        bbox_inches='tight')
    plt.close()
else:
    plot_feature_ranks(df_ranking, metric, feature_names)
    plot_version = 'v1'
    plt.savefig(f"{path_results_in}plot_feature_selection_{classification_type}_dat{data_version}_exp{exp_version}_{metric}_plot{plot_version}.png",
        bbox_inches='tight')
    plt.close()

