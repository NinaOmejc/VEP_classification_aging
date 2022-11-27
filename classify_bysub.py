# This function is called by classify_main.
# The classifier learns and predicts on each subject separately.
# Last updated 18.11.2022, Nina Omejc

# packages
import numpy as np
from scipy import stats, io
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif

def classify_bysub(data, classifier, classifier_name, nSubs, nSamples, nFeatures, nClasses, ID, labels, age, balancing, ma_win, time_switch):

    # initialize overall metrics for classifier evaluation and feature importance over time
    age_compact = np.empty(nSubs, dtype=int)
    probs_total = np.empty((nClasses, nSamples, nSubs))
    acc_total = np.empty((nSamples, nSubs))
    acc_std_total = np.empty((nSamples, nSubs))
    accbal_total = np.empty((nSamples, nSubs))
    accbal_std_total = np.empty((nSamples, nSubs))
    auc_total = np.empty((nSamples, nSubs))
    auc_std_total = np.empty((nSamples, nSubs))
    precision_total = np.empty((nSamples, nSubs))
    recall_total = np.empty((nSamples, nSubs))
    f1_total = np.empty((nSamples, nSubs))
    importance_acc_total = np.empty((nFeatures, nSamples, nSubs))
    importance_acc_std_total = np.empty((nFeatures, nSamples, nSubs))
    importance_accbal_total = np.empty((nFeatures, nSamples, nSubs))
    importance_accbal_std_total = np.empty((nFeatures, nSamples, nSubs))
    importance_auc_total = np.empty((nFeatures, nSamples, nSubs))
    importance_auc_std_total = np.empty((nFeatures, nSamples, nSubs))
    importance_mi_total = np.empty((nFeatures, nSamples, nSubs))

    # get classification for each subject separately
    isub = 0
    for iid in np.unique(ID):
        sub_idx = np.where(ID == iid)
        age_compact[isub] = max(set(age[sub_idx]), key=list(age[sub_idx]).count) # age; 0: young, 1: older
        data_sub = np.squeeze(data[sub_idx, :]) if 'notime' in time_switch else np.squeeze(data[sub_idx, :, :])

        # set empty variables for results of each subject
        probs_sub = np.empty((nClasses, nSamples))
        acc_sub = np.empty(nSamples)
        acc_std_sub = np.empty(nSamples)
        accbal_sub = np.empty(nSamples)
        accbal_std_sub = np.empty(nSamples)
        auc_sub = np.empty(nSamples)
        auc_std_sub = np.empty(nSamples)
        precision_sub = np.empty(nSamples)
        recall_sub = np.empty(nSamples)
        f1_sub = np.empty(nSamples)
        importance_acc_sub = np.empty((nFeatures, nSamples))
        importance_acc_std_sub = np.empty((nFeatures, nSamples))
        importance_accbal_sub = np.empty((nFeatures, nSamples))
        importance_accbal_std_sub = np.empty((nFeatures, nSamples))
        importance_auc_sub = np.empty((nFeatures, nSamples))
        importance_auc_std_sub = np.empty((nFeatures, nSamples))
        importance_mi_sub = np.empty((nFeatures, nSamples))

        # Tail-rolling averasge transform ( not fixed for notime version! )
        if ma_win != 0:
            npad = ((0, 0), (0, 0), (int(ma_win/2), int(ma_win/2)))
            data_sub_pad = np.pad(data_sub, pad_width=npad, mode='edge')
            data_sub_temp = np.cumsum(data_sub_pad, dtype=float, axis=2)
            data_sub_temp[:, :, ma_win:] = data_sub_temp[:, :, ma_win:] - data_sub_temp[:, :, :-ma_win]
            data_sub_ma = data_sub_temp[:, :, ma_win-1:] / ma_win
            data_sub_ma = data_sub_ma[:, :, :nSamples]
            data_sub = np.copy(data_sub_ma)

        # get classification for each timepoint's amplitude separately
        # itime = 0
        for itime in range(nSamples):
            print(f"{classifier_name} | isub: {isub} | subID: {iid} | itime: {itime} / {nSamples - 1}")

            #  initialize evaluation metrics for each time point (averaged over k-folds)
            probs_kfold = np.zeros(nClasses)
            acc_kfold = np.zeros(1)
            acc_std_kfold = []
            accbal_kfold = np.zeros(1)
            accbal_std_kfold = []
            auc_kfold = np.zeros(1)
            auc_std_kfold =[]
            recall_kfold = np.zeros(1)
            precision_kfold = np.zeros(1)
            f1_kfold = np.zeros(1)
            importance_acc_mean_kfold = np.zeros(nFeatures)
            importance_acc_std_kfold = np.zeros(nFeatures)
            importance_accbal_mean_kfold = np.zeros(nFeatures)
            importance_accbal_std_kfold = np.zeros(nFeatures)
            importance_auc_mean_kfold = np.zeros(nFeatures)
            importance_auc_std_kfold = np.zeros(nFeatures)
            importance_mi_scores_kfold = np.zeros(nFeatures)

            # getting X and y
            X = data_sub if 'notime' in time_switch else np.squeeze(data_sub[:, :, itime])
            y = labels[sub_idx]

            # a. split the data into train and test sets using cross-validation
            nFolds = 10
            kfold = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=0)

            for train_ix, test_ix in kfold.split(X, y):
                # print("TRAIN:", train_ix, "TEST:", test_ix)
                X_train_orig, X_test_orig = X[train_ix], X[test_ix]
                y_train_orig, y_test_orig = y[train_ix], y[test_ix]

                # b. combination of over and under sampling using SMOTE or random under/oversampling
                # print(sorted(Counter(y).items()))
                meanSamples = np.mean([Counter(y_train_orig).get(0), Counter(y_train_orig).get(1)], dtype=int)
                rus = RandomUnderSampler(sampling_strategy={0: meanSamples}, random_state=0)
                X_train_temp, y_train_temp = rus.fit_resample(X_train_orig, y_train_orig)

                if balancing == 'smote':
                    overSmote = SMOTE(sampling_strategy={0: meanSamples, 1: meanSamples}, random_state=0)
                    X_train, y_train = overSmote.fit_resample(X_train_temp, y_train_temp)
                    # print(sorted(Counter(y_train).items()))
                elif balancing == 'ros':
                    ros = RandomOverSampler(sampling_strategy={0: meanSamples, 1: meanSamples}, random_state=0)
                    X_train, y_train = ros.fit_resample(X_train_temp, y_train_temp)
                    #print(sorted(Counter(y_train).items()))
                else: # no balancing
                    X_train, y_train = X_train_orig, y_train_orig

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

                # d. choose and train the classifier
                classifier.fit(X_train_norm, y_train)

                # e. test the classifier
                y_pred = classifier.predict(X_test_norm)
                y_pred_prob = classifier.predict_proba(X_test_norm)

                probs_kfold += np.mean(y_pred_prob, axis=0)
                acc_kfold += accuracy_score(y_test_orig, y_pred)
                acc_std_kfold.append(accuracy_score(y_test_orig, y_pred))
                accbal_kfold += balanced_accuracy_score(y_test_orig, y_pred)
                accbal_std_kfold.append(balanced_accuracy_score(y_test_orig, y_pred))
                auc_kfold += roc_auc_score(y_test_orig, y_pred)
                auc_std_kfold.append(roc_auc_score(y_test_orig, y_pred))
                precision_kfold += precision_score(y_test_orig, y_pred, zero_division=0)
                recall_kfold += recall_score(y_test_orig, y_pred, zero_division=0)
                f1_kfold += f1_score(y_test_orig, y_pred, zero_division=0)

                # g. feature importance calculations
                importance_mi_scores_kfold = mutual_info_classif(X_train_norm, y_train)

                importance_acc_kfold = permutation_importance(classifier, X_train_norm, y_train, scoring='accuracy')
                importance_acc_mean_kfold += importance_acc_kfold.importances_mean
                importance_acc_std_kfold += importance_acc_kfold.importances_std

                importance_accbal_kfold = permutation_importance(classifier, X_train_norm, y_train, scoring='balanced_accuracy')
                importance_accbal_mean_kfold += importance_accbal_kfold.importances_mean
                importance_accbal_std_kfold += importance_accbal_kfold.importances_std

                importance_auc_kfold = permutation_importance(classifier, X_train_norm, y_train, scoring='roc_auc')
                importance_auc_mean_kfold += importance_auc_kfold.importances_mean
                importance_auc_std_kfold += importance_auc_kfold.importances_std

            # save an average of kfold scores for a specific subject at specific time point
            probs_sub[:, itime] = np.float64(probs_kfold / nFolds)
            acc_sub[itime] = np.float64(acc_kfold / nFolds)
            acc_std_sub[itime] = np.std(acc_std_kfold)
            accbal_sub[itime] = np.float64(acc_kfold / nFolds)
            accbal_std_sub[itime] = np.std(acc_std_kfold)
            auc_sub[itime] = np.float64(auc_kfold / nFolds)
            auc_std_sub[itime] = np.std(auc_std_kfold)
            recall_sub[itime] = np.float64(recall_kfold / nFolds)
            precision_sub[itime] = np.float64(precision_kfold / nFolds)
            f1_sub[itime] = np.float64(f1_kfold / nFolds)
            importance_acc_sub[:, itime] = importance_acc_mean_kfold / nFolds
            importance_acc_std_sub[:, itime] = importance_acc_std_kfold / nFolds
            importance_accbal_sub[:, itime] = importance_accbal_mean_kfold / nFolds
            importance_accbal_std_sub[:, itime] = importance_accbal_std_kfold / nFolds
            importance_auc_sub[:, itime] = importance_auc_mean_kfold / nFolds
            importance_auc_std_sub[:, itime] = importance_auc_std_kfold / nFolds
            importance_mi_sub[:, itime] = importance_mi_scores_kfold / nFolds

        # save metrics for one subject at all time points
        probs_total[:, :, isub] = probs_sub
        acc_total[:, isub] = acc_sub
        acc_std_total[:, isub] = acc_std_sub
        accbal_total[:, isub] = accbal_sub
        accbal_std_total[:, isub] = accbal_std_sub
        auc_total[:, isub] = auc_sub
        auc_std_total[:, isub] = auc_std_sub
        precision_total[:, isub] = precision_sub
        recall_total[:, isub] = recall_sub
        f1_total[:, isub] = f1_sub
        importance_acc_total[:, :, isub] = importance_acc_sub
        importance_acc_std_total[:, :, isub] = importance_acc_std_sub
        importance_accbal_total[:, :, isub] = importance_accbal_sub
        importance_accbal_std_total[:, :, isub] = importance_accbal_std_sub
        importance_auc_total[:, :, isub] = importance_auc_sub
        importance_auc_std_total[:, :, isub] = importance_auc_std_sub
        importance_mi_total[:, :, isub] = importance_mi_sub

        isub = isub + 1

    results = {
        "probs_total": probs_total,
        "acc_total": acc_total,
        "acc_std_total": acc_std_total,
        "accbal_total": accbal_total,
        "accbal_std_total": accbal_std_total,
        "auc_total": auc_total,
        "auc_std_total": auc_std_total,
        "precision_total": precision_total,
        "recall_total": recall_total,
        "f1_total": f1_total,
        "importance_acc_total": importance_acc_total,
        "importance_acc_std_total": importance_acc_std_total,
        "importance_accbal_total": importance_acc_total,
        "importance_accbal_std_total": importance_acc_std_total,
        "importance_auc_total": importance_auc_total,
        "importance_auc_std_total": importance_auc_std_total,
        "importance_mi_total": importance_mi_total,
    }

    return results, age_compact


