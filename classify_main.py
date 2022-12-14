# This script trains multiple classification algorithms to train the models on ERP dataset.
# ERPs are classified either by subject or all subjects are taken together.
# last updated: 8.12.2022, Nina Omejc

# packages
import os, sys
import time
import pickle
import numpy as np
import itertools
from scipy import io
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from classify_bysub import classify_bysub
from classify_allsubs import classify_allsubs
from classify_allsubsage import classify_allsubsage
from feature_selection import feature_selection_allsubs, feature_selection_bysub
from src.utils import plot_features

# this function is useful for hpc
def set_input(index, classifiers):
    classification_types = ['allsubsage' + time_switch, 'bysub' + time_switch, 'allsubs' + time_switch]
    all_combinations = list(itertools.product(classification_types, classifiers))
    return all_combinations[index]

### SETTINGS ###
time_switch = '_notime' # either "" (for temporal features) ot "_notime" for statistical features of ERP params
data_version = 'v0' + time_switch
exp_version = 'v0'
classifiers_used = ['LDA', 'KNN', 'LR', 'Tree', 'AdaBoost', 'XGB', 'RF', 'SVC_lin', 'SVC_rbf']
balancing = 'smote' # using SMOTE as oversampling technique
ma_win = 0          # define moving average window (not updated)
sFreq = 256         # sampling frequency
classify_onebyone = False
use_cluster = False
plot_features_switch = False

# set classifiers and classification type
if classify_onebyone and use_cluster:
    input_index = int(sys.argv[1])-1
    classification_type, iclf = set_input(input_index, classifiers_used)
elif classify_onebyone and not use_cluster:
    classification_type = sys.argv[1]
    iclf = sys.argv[2]
else:
    classification_type = "bysub" + time_switch
    plot_features_switch = True if "feature_selection" not in classification_type else False

# set target
if "allsubs" in classification_type and not "age" in classification_type:
    target = "stim"
elif "bysub" in classification_type:
    target = "stim"
elif "allsubsage" in classification_type:
    target = "age"

# set paths
path_main = os.getcwd()
path_data = f"{path_main}{os.sep}data_for_classification{os.sep}"
path_out = f"{path_main}{os.sep}results{os.sep}{classification_type}{os.sep}{exp_version}{os.sep}"
os.makedirs(path_out, exist_ok=True)

# import data
data_mat = io.loadmat(file_name=f"{path_data}data_{data_version}.mat")

# reduce the time range due to nans in ersp features
times_orig = data_mat['time_info_orig'][0]
time_startIdx = 35 if "feature" in classification_type else 16 # for feature selection, use only a part of data (only most significant time points)
time_endIdx = 150 if "feature" in classification_type else 239
times = times_orig[time_startIdx:time_endIdx]
nSamples = len(times) if not 'notime' in classification_type else 1

# get data
if 'notime' in classification_type:
    data = data_mat['X']
else:
    data = np.transpose(data_mat['X'][:, time_startIdx:time_endIdx, :], (0, 2, 1))     # from (trials, time points, features) to (trials, features, time points)

nTrials = np.shape(data)[0]
feature_names = data_mat['X_features'][0]
nFeatures = len(feature_names)

# get subject IDs
ID = data_mat['IDs'][:, 0]
ID = np.array([int(i[0]) for i in ID])
age = data_mat['IDs'][:, 1]
age = np.array([i[0] for i in age])
age_coded = np.empty(len(age), dtype=int)
age_coded[age == 'young'] = 0
age_coded[age == 'older'] = 1
nSubs = len(np.unique(ID))
#print(sorted(Counter(ID).items()))

# get labels
yi = np.squeeze(data_mat['yi'])
yi = yi - 1 if 4 in yi else []
label_names = np.squeeze(data_mat['y_cats'])
yDict = dict(OF=0, OR=1, YF=2, YR=3)         # 0: older-freq, 1: older-rare, 2: young-freq, 3: young-rare
#print(sorted(Counter(labels).items()))

# plot features
if plot_features_switch:
    fig_path = path_out
    plot_features(data, times, yi, label_names, feature_names, classification_type, fig_path=fig_path, data_version=data_version)

# define proper labels based on classification type and target
if target == 'stim':
    labels = np.copy(yi)
    labels[np.logical_or(yi == 0, yi == 2)] = 0
    labels[np.logical_or(yi == 1, yi == 3)] = 1
    yDictStim = dict(F=0, R=1) # stimulus target ( 0 = freq, 1 = rare )
    labelIDStim, labelDistributionStim = np.unique(labels, return_counts=True)
    label_names = ['freq', 'rare']
    nClasses = 2
elif target == 'age': # include only rare trials
    trial_labels = np.copy(yi)
    trial_labels[np.logical_or(yi == 0, yi == 2)] = 0
    trial_labels[np.logical_or(yi == 1, yi == 3)] = 1 # stimulus target ( 0 = freq, 1 = rare )
    true_idx = np.logical_not(np.logical_not(trial_labels))
    data = data[true_idx, :] if 'notime' in classification_type else data[true_idx, :, :]
    ID = ID[true_idx]
    age_coded = age_coded[true_idx]
    age = age[true_idx]
    yi = yi[true_idx]
    labels = np.copy(yi)
    labels[np.logical_or(yi == 2, yi == 3)] = 0
    labels[np.logical_or(yi == 0, yi == 1)] = 1
    yDictAge = dict(Y=0, O=1) # age target ( 0 = young, 1 = older )
    labelIDAge, labelDistributionAge = np.unique(labels, return_counts=True)
    nClasses = 2
    label_names = ['young', 'older']
else:
    labels = np.copy(yi)
    labelID, labelDistribution = np.unique(labels, return_counts=True)
    nClasses = 4

del data_mat

### DECODING PART ###
print('Started decoding.') if not "feature" in classification_type else "Started feature selection."
np.random.seed(0)

# define classifiers
xgb_objective = 'binary:logistic' if nClasses == 2 else 'multi:softmax'

classifiers = {
    "LDA": LinearDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "LR": LogisticRegression(class_weight='balanced', max_iter=4000),
    "Tree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(n_estimators=100),
    "XGB": XGBClassifier(objective=xgb_objective, booster='gbtree', eval_metric='auc', max_depth=4, n_estimators=100),
    "RF": RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced'),
    "SVC_lin": SVC(kernel='linear', probability=True, gamma='scale', max_iter=2000),
    "SVC_rbf": SVC(kernel="rbf", probability=True, gamma='scale', max_iter=2000)
}
if classify_onebyone:
    classifiers_used = [iclf]

# classifier_name = classifiers_used[0]
for classifier_name in classifiers_used:
    classifier = classifiers[classifier_name]
    t = time.time()

    if classification_type == "bysub" + time_switch:
        results, age_compact = classify_bysub(data, classifier, classifier_name, nSubs, nSamples, nFeatures, nClasses,
                                              ID, labels, age_coded, balancing, ma_win, time_switch)

    elif classification_type == 'allsubs' + time_switch:
        results, age_compact = classify_allsubs(data, classifier, classifier_name, nSamples, nFeatures, nClasses, labels, balancing, time_switch)

    elif classification_type == 'allsubsage' +time_switch:
        results, age_compact = classify_allsubsage(data, classifier, classifier_name, nSubs, nSamples, nFeatures, nClasses,
                                              ID, labels, age_coded, balancing, ma_win, time_switch)

    elif "feature_selection_allsubs" in classification_type:
        results, age_compact = feature_selection_allsubs(data, classifier, classifier_name, nSubs, nSamples, nFeatures, nClasses,
                                              ID, labels, age_coded, balancing, ma_win, target, time_switch)

    elif "feature_selection_bysub" in classification_type:
        results, age_compact = feature_selection_bysub(data, classifier, classifier_name, nSubs, nSamples, nFeatures, nClasses,
                                              ID, labels, age_coded, balancing, ma_win, target, time_switch)

    duration = time.time() - t

    # save results
    with open(f"{path_out}classificaton_results_{classification_type}_dat{data_version}_exp{exp_version}"
              f"_{balancing}_{target}_ma{ma_win}_{classifier_name}.pkl", 'wb') as f:
        pickle.dump([results, ID, age, times, labels, duration, age_compact], f, protocol=-1)


# age_compact = np.hstack((np.ones(43, dtype="int"), np.zeros(27, dtype="int")))
# save variables needed for further analysis
general_info = {
    "classification_type": classification_type,
    "balancing": balancing,
    "classifiers_used": classifiers_used,
    "feature_names": feature_names,
    "label_names": label_names,
    "target": target,
    "ma_win": ma_win,
    "times": times,
    "subID": ID,
    "labels": labels,
    "age": age,
    "age_coded": age_coded,
    "age_compact": age_compact,
}

with open(f"{path_out}classificaton_results_{classification_type}_dat{data_version}_exp{exp_version}_"
          f"{balancing}_{target}_ma{ma_win}_general_info.pkl", 'wb') as f:
    pickle.dump(general_info, f, protocol=-1)