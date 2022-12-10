# Uses three types of feature selection algorithms (RFE, SequentialFeatureSelector & mutual_info_classif).
# It has to be manually switched between them.
# Script is called by classify_main, with specifying classification_type to either feature_selection_bysub or feature_selection_allsubs
# Last updated: 8.12.2022

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE, SequentialFeatureSelector, mutual_info_classif


def feature_selection_allsubs(data, classifier, classifier_name, nSubs, nSamples, nFeatures, nClasses,
                      ID, labels, age_coded, balancing, ma_win, target, time_switch):

    # initialize overall metrics for classifier evaluation and feature importance over time
    #rfe_ranking_total = np.empty((nFeatures, nSamples))
    #sfs_support_total = np.empty((nFeatures, nSamples))
    mi_total = np.empty((nFeatures, nSamples))

    # get classification for each timepoint's amplitude separately
    # itime = 0
    for itime in range(nSamples):
        print(f"{classifier_name} | allsubs | {target} | itime: {itime} / {nSamples - 1}")

        # initialize evaluation metrics for each time point (averaged over k-folds)
        # rfe_ranking_kfold = np.zeros(nFeatures)
        # sfs_support_kfold = np.zeros(nFeatures)
        mi_kfold = np.zeros(nFeatures)

        # getting X and y
        X = data if 'notime' in time_switch else np.squeeze(data[:, :, itime])
        y = np.copy(labels)

        # a. split the data into train and test sets using cross-validation
        nFolds = 10
        kfold = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=1)

        for train_ix, test_ix in kfold.split(X, y):
            # print("TRAIN:", train_ix, "TEST:", test_ix)
            X_train_orig, X_test_orig = X[train_ix], X[test_ix]
            y_train_orig, y_test_orig = y[train_ix], y[test_ix]

            # b. combination of over and under sampling using SMOTE or random under/oversampling
            underidx = 0 if target == "stim" else 1
            overidx = 1 if underidx == 0 else 1
            meanSamples = np.mean([Counter(y_train_orig).get(0), Counter(y_train_orig).get(1)], dtype=int)
            rus = RandomUnderSampler(sampling_strategy={underidx: meanSamples}, random_state=0)
            X_train_temp, y_train_temp = rus.fit_resample(X_train_orig, y_train_orig)

            if balancing == 'smote':
                overSmote = SMOTE(sampling_strategy={overidx: meanSamples}, random_state=0)
                X_train, y_train = overSmote.fit_resample(X_train_temp, y_train_temp)
            elif balancing == 'ros':
                ros = RandomOverSampler(sampling_strategy={overidx: meanSamples}, random_state=0)
                X_train, y_train = ros.fit_resample(X_train_temp, y_train_temp)
            else:  # no balancing
                X_train, y_train = X_train_orig, y_train_orig
            # print(sorted(Counter(y_train).items()))

            # shuffle training data after they are sorted by under and over samplers:
            rng = np.random.default_rng()
            shuffle_indices = np.arange(len(y_train))
            rng.shuffle(shuffle_indices)
            X_train = X_train[shuffle_indices, :]
            y_train = y_train[shuffle_indices]

            # c. normalize data
            scaler = MinMaxScaler().fit(X_train)
            X_train_norm = scaler.transform(X_train)
            X_test_norm = scaler.transform(X_test_orig)

            # define feature selectors
            #sfs = SequentialFeatureSelector(classifier, n_features_to_select=8, scoring="roc_auc", direction="forward")
            #rfe = RFE(classifier, n_features_to_select=10, step=1)

            # d. choose and train the classifier
            #rfe.fit(X_train_norm, y_train)
            #sfs.fit(X_train_norm, y_train)
            mi = mutual_info_classif(X_train_norm, y_train)

            # e. get selected features
            #rfe_ranking_kfold += rfe.ranking_
            #sfs_support_kfold += sfs.get_support().astype("int")
            mi_kfold += mi

        # save an average of kfold scores for specific time point
        #rfe_ranking_total[:, itime] = np.float(rfe_ranking_kfold / nFolds)
        #sfs_support_total[:, itime] = np.float(sfs_support_kfold / nFolds)
        mi_total[:, itime] = mi_kfold

    results = {
      #  "rfe_ranking_total": rfe_ranking_total,
      #  "sfs_support_total": sfs_support_total,
        "mi_total": mi_total,
    }

    return results, []


def feature_selection_bysub(data, classifier, classifier_name, nSubs, nSamples, nFeatures, nClasses,
                      ID, labels, age_coded, balancing, ma_win, target, time_switch):

    # initialize overall metrics for classifier evaluation and feature importance over time
    age_compact = np.empty(nSubs, dtype=int)
    # rfe_ranking_total = np.empty((nFeatures, nSamples, nSubs))
    # sfs_support_total = np.empty((nFeatures, nSamples, nSubs))
    mi_total = np.empty((nFeatures, nSamples, nSubs))

    # get classification for each subject separately
    isub = 0
    for iid in np.unique(ID):
        sub_idx = np.where(ID == iid)
        age_compact[isub] = max(set(age_coded[sub_idx]), key=list(age_coded[sub_idx]).count)  # age; 0: young, 1: older
        data_sub = np.squeeze(data[sub_idx, :]) if 'notime' in time_switch else np.squeeze(data[sub_idx, :, :])

        # rfe_ranking_sub = np.empty((nFeatures, nSamples))
        # sfs_support_sub = np.empty((nFeatures, nSamples))
        mi_sub = np.empty((nFeatures, nSamples))

        # get classification for each timepoint's amplitude separately
        # itime = 0
        for itime in range(nSamples):
            print(f"{classifier_name} | isub: {isub} | subID: {iid} | itime: {itime} / {nSamples - 1}")

            # initialize evaluation metrics for each time point (averaged over k-folds)
            # rfe_ranking_kfold = np.zeros(nFeatures)
            # sfs_support_kfold = np.zeros(nFeatures)
            mi_kfold = np.zeros(nFeatures)

            # getting X and y
            X = data_sub if 'notime' in time_switch else np.squeeze(data_sub[:, :, itime])
            y = labels[sub_idx]

            # a. split the data into train and test sets using cross-validation
            nFolds = 10
            kfold = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=1)

            for train_ix, test_ix in kfold.split(X, y):
                # print("TRAIN:", train_ix, "TEST:", test_ix)
                X_train_orig, X_test_orig = X[train_ix], X[test_ix]
                y_train_orig, y_test_orig = y[train_ix], y[test_ix]

                # b. combination of over and under sampling using SMOTE or random under/oversampling
                meanSamples = np.mean([Counter(y_train_orig).get(0), Counter(y_train_orig).get(1)], dtype=int)
                rus = RandomUnderSampler(sampling_strategy={0: meanSamples}, random_state=0)
                X_train_temp, y_train_temp = rus.fit_resample(X_train_orig, y_train_orig)

                if balancing == 'smote':
                    overSmote = SMOTE(sampling_strategy={1: meanSamples}, random_state=0)
                    X_train, y_train = overSmote.fit_resample(X_train_temp, y_train_temp)
                elif balancing == 'ros':
                    ros = RandomOverSampler(sampling_strategy={1: meanSamples}, random_state=0)
                    X_train, y_train = ros.fit_resample(X_train_temp, y_train_temp)
                else:  # no balancing
                    X_train, y_train = X_train_orig, y_train_orig
                # print(sorted(Counter(y_train).items()))

                # shuffle training data after they are sorted by under and over samplers:
                rng = np.random.default_rng()
                shuffle_indices = np.arange(len(y_train))
                rng.shuffle(shuffle_indices)
                X_train = X_train[shuffle_indices, :]
                y_train = y_train[shuffle_indices]

                # c. normalize data
                scaler = MinMaxScaler().fit(X_train)
                X_train_norm = scaler.transform(X_train)
                X_test_norm = scaler.transform(X_test_orig)

                # selectors
                #sfs = SequentialFeatureSelector(classifier, n_features_to_select=8, scoring="roc_auc", direction="forward")
                #rfe = RFE(classifier, n_features_to_select=10, step=1)

                #rfe.fit(X_train_norm, y_train)
                #sfs.fit(X_train_norm, y_train)
                mi = mutual_info_classif(X_train_norm, y_train)

                # e. get selected features
                #rfe_ranking_kfold += rfe.ranking_
                #sfs_support_kfold += sfs.get_support().astype("int")
                mi_kfold += mi

            #rfe_ranking_sub[:, itime] = np.float64(rfe_ranking_kfold / nFolds)
            #sfs_support_sub[:, itime] = np.float64(sfs_support_kfold / nFolds)
            mi_sub[:, itime] = mi_kfold / nFolds

        # save an average of kfold scores for specific time point
        #rfe_ranking_total[:, :, isub] = rfe_ranking_sub
        #sfs_support_total[:, :, isub] = sfs_support_sub
        mi_total[:, :, isub] = mi_sub
        isub = isub + 1

    results = {
     #   "rfe_ranking_total": rfe_ranking_total,
     #   "sfs_support_total": sfs_support_total,
        "mi_total": mi_total,
    }

    return results, age_compact