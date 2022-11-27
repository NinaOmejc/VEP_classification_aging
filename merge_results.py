import os
import pickle
import numpy as np
import pandas as pd

### IMPORT STUDY SETTINGS
time_switch = '_notime'
classification_type = 'bysub' + time_switch
data_version = 'v1' + time_switch
exp_version = 'v2'
balancing = 'smote'
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

path_main = f"N:{os.sep}SloMoBIL{os.sep}classification_paper{os.sep}"
#path_main = f"D:{os.sep}Experiments{os.sep}erp_classification_study{os.sep}"
path_results_in = f"{path_main}results{os.sep}{classification_type}{os.sep}{exp_version}{os.sep}"

general_info_filename = f"{path_results_in}classificaton_results_{classification_type}_dat{data_version}_exp{exp_version}_{balancing}_{target}_ma{ma_win}_general_info.pkl"
with open(general_info_filename, 'rb') as f:
    general_info = pickle.load(f)

feature_names = [general_info["feature_names"][i][0] for i in range(len(general_info["feature_names"]))]
label_names = [general_info["label_names"][i][0] for i in range(len(general_info["label_names"]))]
times = general_info["times"] if time_switch == '' else 0
subID = general_info["subID"]
age_compact = general_info["age_compact"]

nSamples = len(times) if time_switch == '' else 1
nSubj = len(np.unique(subID)) if classification_type == 'bysub' + time_switch else 1
nClassifiers = len(classifier_names)
nFeatures = len(feature_names)
nClasses = len(label_names)

# set up empty pandas dataframe to collect results
bysub_columns = ['sub', 'age'] if classification_type == 'bysub' + time_switch else []
df_probs = pd.DataFrame(columns=['time'] + bysub_columns + ['model', 'class', 'Probabilities'])
df_metrics = pd.DataFrame(columns=['time'] + bysub_columns + ['model'] + metric_names)
df_importances = pd.DataFrame(columns=['time'] + bysub_columns + ['feature', 'model'] + importance_metric_names)
df_duration = pd.DataFrame(columns=['model', 'Duration'])

# add time, subj, age & features / class info to dataframes in a melted format
df_duration['model'] = classifier_names

df_probs['model'] = np.tile(np.repeat(classifier_names, nSamples * nClasses), nSubj)
df_probs['class'] = np.repeat(np.tile(label_names, nSubj*nClassifiers), nSamples)
df_probs['time'] = np.tile(times, nClasses*nClassifiers*nSubj)

df_metrics['model'] = np.tile(np.repeat(classifier_names, nSamples), nSubj)
df_metrics['time'] = np.tile(times, nClassifiers*nSubj)

df_importances['model'] = np.tile(np.repeat(classifier_names, nSamples * nFeatures), nSubj)
df_importances['feature'] = np.repeat(np.tile(feature_names, nSubj*nClassifiers), nSamples)
df_importances['time'] = np.tile(times, nFeatures*nClassifiers*nSubj)

df_probs["model"] = df_probs["model"].astype("category")
df_probs["class"] = df_probs["class"].astype("category")
df_metrics["model"] = df_metrics["model"].astype("category")
df_importances["model"] = df_importances["model"].astype("category")
df_importances["feature"] = df_importances["feature"].astype("category")

if classification_type == 'bysub' + time_switch:
    df_probs['age'] = np.repeat(age_compact, nClassifiers*nClasses*nSamples)
    df_probs['sub'] = np.repeat(np.unique(subID), nClassifiers*nClasses*nSamples)
    df_metrics['age'] = np.repeat(age_compact, nClassifiers*nSamples)
    df_metrics['sub'] = np.repeat(np.unique(subID), nClassifiers*nSamples)
    df_importances['age'] = np.repeat(age_compact, nClassifiers * nFeatures * nSamples)
    df_importances['sub'] = np.repeat(np.unique(subID), nClassifiers * nFeatures * nSamples)

    df_probs["sub"] = df_probs["sub"].astype("category")
    df_probs["age"] = df_probs["age"].astype("category")
    df_metrics["sub"] = df_metrics["sub"].astype("category")
    df_metrics["age"] = df_metrics["age"].astype("category")
    df_importances["sub"] = df_importances["sub"].astype("category")
    df_importances["age"] = df_importances["age"].astype("category")

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
    transpose_indices = [1, 0, 2] if classification_type == 'bysub' + time_switch else [1, 0]
    df_duration.loc[df_duration["model"] == clfName, "Duration"] = duration
    df_probs.loc[df_probs["model"] == clfName, "Probabilities"] = results["probs_total"].transpose(transpose_indices).flatten('F')

    df_metrics.loc[df_metrics["model"] == clfName, metric_names[0]] = results["acc_total"].flatten('F')
    df_metrics.loc[df_metrics["model"] == clfName, metric_names[1]] = results["acc_std_total"].flatten('F')
    df_metrics.loc[df_metrics["model"] == clfName, metric_names[2]] = results["accbal_total"].flatten('F')
    df_metrics.loc[df_metrics["model"] == clfName, metric_names[3]] = results["accbal_std_total"].flatten('F')
    df_metrics.loc[df_metrics["model"] == clfName, metric_names[4]] = results["auc_total"].flatten('F')
    df_metrics.loc[df_metrics["model"] == clfName, metric_names[5]] = results["auc_std_total"].flatten('F')
    df_metrics.loc[df_metrics["model"] == clfName, metric_names[6]] = results["precision_total"].flatten('F')
    df_metrics.loc[df_metrics["model"] == clfName, metric_names[7]] = results["recall_total"].flatten('F')
    df_metrics.loc[df_metrics["model"] == clfName, metric_names[8]] = results["f1_total"].flatten('F')

    df_importances.loc[df_importances["model"] == clfName, importance_metric_names[0]] = results["importance_acc_total"].transpose(transpose_indices).flatten('F')
    df_importances.loc[df_importances["model"] == clfName, importance_metric_names[1]] = results["importance_acc_std_total"].transpose(transpose_indices).flatten('F')
    df_importances.loc[df_importances["model"] == clfName, importance_metric_names[2]] = results["importance_accbal_total"].transpose(transpose_indices).flatten('F')
    df_importances.loc[df_importances["model"] == clfName, importance_metric_names[3]] = results["importance_accbal_std_total"].transpose(transpose_indices).flatten('F')
    df_importances.loc[df_importances["model"] == clfName, importance_metric_names[4]] = results["importance_auc_total"].transpose(transpose_indices).flatten('F')
    df_importances.loc[df_importances["model"] == clfName, importance_metric_names[5]] = results["importance_acc_std_total"].transpose(transpose_indices).flatten('F')
    df_importances.loc[df_importances["model"] == clfName, importance_metric_names[6]] = results["importance_mi_total"].transpose(transpose_indices).flatten('F')

# SAVE RESULTS INTO COMMON MERGED PICKLE FILE
with open(f"{path_results_in}merged_classificaton_results_dat{data_version}_exp{exp_version}_{balancing}_{target}_ma{ma_win}.pkl", 'wb') as f:
    pickle.dump([general_info, df_duration, df_probs, df_metrics, df_importances], f, protocol=-1)


