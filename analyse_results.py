import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
plt.interactive(False)
from src.utils import calculate_cd_diagram
from src.analyse_results_plotting import plot_metrics_averaged, plot_imetric_agestats,\
    plot_probs, plot_feature_importance, plot_feature_importance_clfavg, plot_cd_diagram_auroc, \
    plot_individual_max, plot_timepoints_at_individual_max

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

### GENERAL SETTINGS
time_switch = '_notime'
data_version = 'v0' + time_switch
exp_version = 'v0'
analysis_version = 'v0'
classification_type = 'bysub' + time_switch
balancing = "smote"
ma_win = 0

classifier_names = ['LDA', 'KNN', 'LR', 'Tree', 'AdaBoost', 'XGB', 'RF', 'SVC_lin', 'SVC_rbf']
metric_names = ['Accuracy', 'Accuracy_Std', 'Balanced_Accuracy', 'Balanced_Accuracy_Std',
                'AUROC', 'AUROC_Std', 'Precision', 'Recall', 'F1']
importance_metric_names = ['Importance_Acc', 'Importance_Acc_Std', 'Importance_Balanced_Acc',
                           'Importance_Balanced_Acc_Std', 'Importance_AUROC', 'Importance_AUROC_Std', 'Importance_MI']

if classification_type == "allsubs" + time_switch:
    target = "stim&age"
elif classification_type == "bysub" + time_switch:
    target = "stim"
elif classification_type == "allsubsage" + time_switch:
    target = "age"

path_main = os.getcwd()
path_results_in = f"{path_main}\\results\\{classification_type}\\{exp_version}\\"
path_analysis_out = f"{path_results_in}analysis\\{analysis_version}\\"
os.makedirs(path_analysis_out, exist_ok=True)

### IMPORT RESULTS
merged_results_filename = f"merged_classificaton_results_dat{data_version}_exp{exp_version}_{balancing}_{target}_ma{ma_win}.pkl"
with open(f"{path_results_in}{merged_results_filename}", 'rb') as f:
    [general_info, df_duration, df_probs, df_metrics, df_importances] = pickle.load(f)

times = general_info["times"] if time_switch == '' else 0
nSamples = 1 if time_switch == '_notime' else len(times)
subID = general_info["subID"]
feature_names = [general_info["feature_names"][i][0] for i in range(len(general_info["feature_names"]))]
label_names = [general_info["label_names"][i] for i in range(len(general_info["label_names"]))]

for imetric in metric_names:
    df_metrics[imetric] = df_metrics[imetric].astype('float')

if 'notime' in time_switch:
    df_importances["feature"].cat.reorder_categories(feature_names)
    df_importances["feature"].cat.rename_categories([feature_names[i][2:].replace('_', ': ') for i in range(len(feature_names))], inplace=True)
    feature_names = df_importances["feature"].cat.categories

if 'bysub' in classification_type:
    df_metrics["age"] = df_metrics["age"].astype("int").astype("category")
    df_probs["age"] = df_probs["age"].astype("int").astype("category")
    df_importances["age"] = df_importances["age"].astype("int").astype("category")

# GENERAL STATISTICS
# number of trials per subject, mean and std
c = dict(Counter(subID).items())
c_vals = list(c.values())
c_vals_mean = np.mean(c_vals)
c_vals_std = np.std(c_vals)

# number of subjects in each age group
age_count = dict(Counter(general_info["age_compact"]).items())

# number of trials for each stimulus type (before sampling)
labels_count = dict(Counter(general_info["labels"]).items())

# check max
a = df_metrics.groupby(["time", "model"], as_index=False).mean()
a.loc[a["model"] == "LDA", "AUROC"].argmax()

a = df_metrics.groupby(["time", "model"], as_index=False).mean()
a.loc[a["model"] == "LDA", "AUROC"].max()

#########################################################################
                   ### PROBS PLOT ###
#########################################################################

clfs_included = classifier_names[:]
plot_probs(df_probs, clfs_included, classification_type)
plot_version = 'v2'
plt.savefig(f"{path_analysis_out}plot_probs_byclf_{classification_type}_dat{data_version}_exp{exp_version}_ana{analysis_version}_plot{plot_version}.png",
            bbox_inches='tight')
plt.close('all')

#########################################################################
                   ### EVALUATION METRICS PLOT ###
#########################################################################

clfs_included = classifier_names[:]
metrics_included = ['Accuracy', 'AUROC', 'Precision', 'Recall', 'F1']

plot_metrics_averaged(df_metrics, clfs_included, metrics_included, classification_type)

plot_version = 'v2'
plt.savefig(f"{path_analysis_out}plot_metrics_clfavg_{classification_type}_dat{data_version}_exp{exp_version}_ana{analysis_version}_plot{plot_version}.png",
            bbox_inches='tight')
plt.close('all')


#########################################################################
                        ### FEATURE IMPORTANCE PLOT  ###
                    ### grouped by age, individual clfs ###
#########################################################################

clfs_included = classifier_names[:] #["LDA", "RF", "KNN"]
importance_metric_included = importance_metric_names[4]

plot_feature_importance(df_importances, importance_metric_included, clfs_included, classification_type)

plot_version = 'v2'
plt.savefig(f"{path_analysis_out}plot_feature_{importance_metric_included}_byclf_{classification_type}_dat{data_version}_exp{exp_version}"
        f"_ana{analysis_version}_plot{plot_version}.png", bbox_inches='tight')
plt.close('all')

#########################################################################
                        ### FEATURE IMPORTANCE PLOT  ###
                          ### grouped by age if bysub ##
                               ## averaged clfs ###
#########################################################################

clfs_included = classifier_names[:] #["LDA", "RF", "KNN"]
importance_metric_included = importance_metric_names[4]

plot_feature_importance_clfavg(df_importances, importance_metric_included, clfs_included, classification_type)

plot_version = 'v2'
plt.savefig(f"{path_analysis_out}plot_feature_{importance_metric_included}_clfavg_{classification_type}_dat{data_version}_exp{exp_version}"
        f"_ana{analysis_version}_plot{plot_version}.png", bbox_inches='tight')
plt.close('all')

#########################################################################
                   ### EVALUATION METRICS PLOT ###
          ### grouped by age but only one metric; AUROC ###
                   ### add statistics as well! ###
#########################################################################
if "bysub" in classification_type:
    clfs_included = classifier_names[:]  # ['LDA', 'RF', 'KNN']
    metrics_included = ['AUROC']

    plot_imetric_agestats(df_metrics, clfs_included, metrics_included, age_count, classification_type)
    plot_version = 'v2'
    plt.savefig(f"{path_analysis_out}plot_{metrics_included[0]}_byclf_agestats_{classification_type}_dat{data_version}_exp{exp_version}_ana{analysis_version}_plot{plot_version}.png",
                bbox_inches='tight')
    plt.close()

#########################################################################
                          ### CD plot ###
                            ## AUROC ##
#########################################################################
if "bysub" in classification_type:
    metrics_included = 'AUROC'
    path_cd_true = f'{path_results_in}cd_{metrics_included}_true\\'
    path_cd_ranks = f'{path_results_in}cd_{metrics_included}_ranks\\'

    clf_dict = {'LDA': 0, 'LR': 1, 'SVC_lin': 2, 'SVC_rbf': 3, 'RF': 4,
                'XGB': 5, 'KNN': 6, 'AdaBoost': 7, 'Tree': 8}

    calculate_cd_diagram(df_metrics, metrics_included, path_cd_ranks, path_cd_true)

    if not "notime" in classification_type:
        plot_cd_diagram_auroc(clf_dict, times, path_cd_true, path_cd_ranks)
        plot_version = 'v3'
        plt.savefig(f"{path_analysis_out}plot_cd_diagram_{metrics_included}_{classification_type}_dat{data_version}_exp{exp_version}_ana{analysis_version}_plot{plot_version}.png",
                    bbox_inches="tight")
        plt.close('all')

#########################################################################
                ### Plot max vrednosti posameznih udeležencev ###
                           ### grouped by age ###
#########################################################################
if "bysub" in classification_type:
    metric_included = "AUROC"
    clfs_included = classifier_names[:] #["LDA", "RF", "KNN"]

    plot_individual_max(df_metrics, clfs_included, metric_included)

    plot_version = 'v2'
    plt.savefig(f"{path_analysis_out}plot_maxvaldistr_byclf_{classification_type}_dat{data_version}_exp{exp_version}"
            f"_ana{analysis_version}_plot{plot_version}.png", bbox_inches='tight')
    plt.close('all')

#########################################################################
     ### Plot time points for max vrednosti posameznih udeležencev ###
                    ### grouped by age ###
#########################################################################

if "bysub" in classification_type and "notime" not in classification_type:

    metric_included = "AUROC"
    clfs_included = classifier_names[:]

    plot_timepoints_at_individual_max(df_metrics, clfs_included, metric_included)

    plot_version = 'v2'
    plt.savefig(f"{path_analysis_out}plot_timepointsatmax_byclf_{classification_type}_dat{data_version}_exp{exp_version}"
            f"_ana{analysis_version}_plot{plot_version}.png", bbox_inches='tight')
    plt.close('all')

