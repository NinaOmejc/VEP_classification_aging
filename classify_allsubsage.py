# This function is called by classify_main.
# The classifier learns and predicts on all subjects separately but specific
# for target=age group.
# Last updated 18.11.2022, Nina Omejc

# packages
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif


def classify_allsubsage(data, classifier, classifier_name, nSubs, nSamples, nFeatures, nClasses, ID, labels, age_coded, balancing, ma_win, time_switch):

    # initialize overall metrics for classifier evaluation and feature importance over time
    probs_total = np.empty((nClasses, nSamples))
    cm_total = np.empty((nClasses*2, nSamples))
    acc_total = np.empty(nSamples)
    acc_std_total = np.empty(nSamples)
    accbal_total = np.empty(nSamples)
    accbal_std_total = np.empty(nSamples)
    auc_total = np.empty(nSamples)
    auc_std_total = np.empty(nSamples)
    precision_total = np.empty(nSamples)
    recall_total = np.empty(nSamples)
    f1_total = np.empty(nSamples)
    importance_acc_total = np.empty((nFeatures, nSamples))
    importance_acc_std_total = np.empty((nFeatures, nSamples))
    importance_accbal_total = np.empty((nFeatures, nSamples))
    importance_accbal_std_total = np.empty((nFeatures, nSamples))
    importance_auc_total = np.empty((nFeatures, nSamples))
    importance_auc_std_total = np.empty((nFeatures, nSamples))
    importance_mi_total = np.empty((nFeatures, nSamples))

    # get classification for each timepoint's amplitude separately
    # itime = 0
    for itime in range(nSamples):
        print(f"{classifier_name} | allsubsage | itime: {itime} / {nSamples-1}")

        # initialize evaluation metrics for each time point (averaged over k-folds)
        probs_kfold = np.zeros(nClasses)
        cm_kfold = np.zeros(nClasses*2)
        acc_kfold = np.zeros(1)
        acc_std_kfold = []
        accbal_kfold = np.zeros(1)
        accbal_std_kfold = []
        auc_kfold = np.zeros(1)
        auc_std_kfold = []
        recall_kfold = np.zeros(1)
        precision_kfold = np.zeros(1)
        f1_kfold = np.zeros(1)
        importance_acc_mean_kfold = np.zeros(nFeatures)
        importance_acc_std_kfold = np.zeros(nFeatures)
        importance_auc_mean_kfold = np.zeros(nFeatures)
        importance_auc_std_kfold = np.zeros(nFeatures)
        importance_accbal_mean_kfold = np.zeros(nFeatures)
        importance_accbal_std_kfold = np.zeros(nFeatures)
        importance_mi_scores_kfold = np.zeros(nFeatures)

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
            meanSamples = np.mean([Counter(y_train_orig).get(0), Counter(y_train_orig).get(1)], dtype=int)
            rus = RandomUnderSampler(sampling_strategy={1: meanSamples}, random_state=0)
            X_train_temp, y_train_temp = rus.fit_resample(X_train_orig, y_train_orig)

            if balancing == 'smote':
                overSmote = SMOTE(sampling_strategy={0: meanSamples,}, random_state=0)
                X_train, y_train = overSmote.fit_resample(X_train_temp, y_train_temp)
            elif balancing == 'ros':
                ros = RandomOverSampler(sampling_strategy={0: meanSamples}, random_state=0)
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

            # d. choose and train the classifier
            classifier.fit(X_train_norm, y_train)

            # e. test the classifier
            y_pred = classifier.predict(X_test_norm)
            y_pred_prob = classifier.predict_proba(X_test_norm)

            # f. evaluate classifier
            probs_kfold += np.mean(y_pred_prob, axis=0)
            cm_kfold += confusion_matrix(y_test_orig, y_pred).reshape(nClasses * 2)
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

            importance_auc_kfold = permutation_importance(classifier, X_train_norm, y_train, scoring='roc_auc_ovr')
            importance_auc_mean_kfold += importance_auc_kfold.importances_mean
            importance_auc_std_kfold += importance_auc_kfold.importances_std

        # save an average of kfold scores for specific time point
        probs_total[:, itime] = np.float64(probs_kfold / nFolds)
        cm_total[:, itime] = np.float64(cm_kfold / nFolds)
        acc_total[itime] = np.float64(acc_kfold / nFolds)
        acc_std_total[itime] = np.std(acc_std_kfold)
        accbal_total[itime] = np.float64(accbal_kfold / nFolds)
        accbal_std_total[itime] = np.std(accbal_std_kfold)
        auc_total[itime] = np.float64(auc_kfold / nFolds)
        auc_std_total[itime] = np.std(auc_std_kfold)
        recall_total[itime] = np.float64(recall_kfold / nFolds)
        precision_total[itime] = np.float64(precision_kfold / nFolds)
        f1_total[itime] = np.float64(f1_kfold / nFolds)
        importance_acc_total[:, itime] = np.float64(importance_acc_mean_kfold / nFolds)
        importance_acc_std_total[:, itime] = np.float64(importance_acc_std_kfold / nFolds)
        importance_accbal_total[:, itime] = np.float64(importance_accbal_mean_kfold / nFolds)
        importance_accbal_std_total[:, itime] = np.float64(importance_accbal_std_kfold / nFolds)
        importance_auc_total[:, itime] = np.float64(importance_auc_mean_kfold / nFolds)
        importance_auc_std_total[:, itime] = np.float64(importance_auc_std_kfold / nFolds)
        importance_mi_total[:, itime] = np.float64(importance_mi_scores_kfold / nFolds)

    results = {
        "probs_total": probs_total,
        "cm_total": cm_total,
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
        "importance_accbal_total": importance_accbal_total,
        "importance_accbal_std_total": importance_accbal_std_total,
        "importance_auc_total": importance_auc_total,
        "importance_auc_std_total": importance_auc_std_total,
        "importance_mi_total": importance_mi_total,
    }

    return results, []