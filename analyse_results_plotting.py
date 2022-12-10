import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
plt.interactive(False)
import scipy as sp
import numpy as np
from collections import Counter
from cycler import cycler
import scipy.io
from mne.stats import permutation_cluster_test
import pandas as pd
import pickle
from scipy.signal import savgol_filter
import seaborn as sns
from cd_diagram import draw_cd_diagram
from scipy.stats import ttest_ind
#from scikit_posthocs import posthoc_dunn

plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.titlesize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['lines.linewidth'] = 1

xticks = np.arange(-200, 800, 200)
xlim = [-120, 720]

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:brown', 'tab:gray']
color_cycle = cycler(color=colors)  # ])
# marker_cycle = cycler(marker=['o', 'v'])
# mc_cycle = color_cycle * marker_cycle
plt.rcParams['legend.labelcolor'] = 'black'
plt.rcParams['axes.prop_cycle'] = color_cycle
# plt.rc('axes', prop_cycle=mc_cycle)

def plot_probs(df_probs, clfs_included, classification_type):

    np.random.seed(0) # set seed for reproducibility in permutation statistics
    classes = np.unique(df_probs["class"])
    nCols = len(clfs_included)
    df_probs = df_probs.dropna()

    if "notime" in classification_type:
        plt.figure()
        sns.catplot(data=df_probs, x="class", y="Probabilities", hue="model", hue_order=clfs_included, kind="point",
                    legend=False)
        plt.xticks(ticks=[0, 1], labels=['young', 'older'])
        plt.legend(ncol=2, loc="upper right", frameon=True)
    else:
        fig, axs = plt.subplots(1, nCols, figsize=(8, 6))
        plt.subplots_adjust(wspace=0, hspace=0)
        for iax, ax in enumerate(axs):
            legend_switch = True if iax == 0 else False
            sns.lineplot(x="time", y="Probabilities", hue="class", data=df_probs[df_probs['model'] == clfs_included[iax]],
                         legend=legend_switch, ax=axs[iax], palette=sns.color_palette()[:len(classes)], lw='0.7')
            axs[iax].set_title(clfs_included[iax])
            axs[iax].set_xticks(xticks)
            axs[iax].set_xlim(xlim)
            axs[iax].spines['top'].set_visible(False)
            axs[iax].spines['right'].set_visible(False)
            axs[iax].tick_params(axis='both', which='major', direction="in")
            axs[iax].axvline(x=0, color='gray', linestyle='--', linewidth=0.25)
            axs[iax].set_xlabel("")
            axs[iax].set_ylabel("")
            if classification_type == "allsubs":
                axs[iax].set_ylim([0.1, 0.40])
                axs[iax].set_yticks(np.arange(0.1, 0.41, 0.1))
                axs[iax].set_yticklabels(['0.1', '0.2', '0.3', '0.4'])
            elif classification_type == "bysub":
                axs[iax].set_ylim([0.2, 0.8])
                axs[iax].set_yticks(np.arange(0.2, 0.8, 0.1))
                axs[iax].set_yticklabels(['0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8'])

            if iax == 0:
                axs[iax].set_xticklabels(['', '0', '200', '400', '600'])
                axs[iax].set_xlabel('Time / ms')
                axs[iax].legend(ncol=2, loc="upper right", frameon=True)
            else:
                axs[iax].axes.xaxis.set_ticklabels([])
                axs[iax].axes.yaxis.set_ticklabels([])
    plt.show()

def plot_metrics_averaged(df_metrics, clfs_included, metrics_included, classification_type):

    if "_notime" in classification_type:
        df_metrics_avg = df_metrics.drop("time", axis=1).groupby("model", as_index=False)[metrics_included].mean()
    else:
        df_metrics_avg = df_metrics.groupby(["time", "model"], as_index=False)[metrics_included].mean()

    nMetrics = len(metrics_included)
    nRows = 2
    nCols = nMetrics // nRows if nMetrics % nRows == 0 else nMetrics // nRows + 1
    fig, axs = plt.subplots(nRows, nCols, figsize=(9, 4))
    plt.subplots_adjust(wspace=0, hspace=0.01)
    before = True
    # im = 0
    for im in range(nRows * nCols):
        iax = im if before == True else im - 1
        if [im // nCols, im % nCols] == [0, nCols - 1]:
            axs[im // nCols, im % nCols].axis('off')
            before = False
            continue
        legend_switch = True if [im // nCols, im % nCols] == [0, nCols - 2] else False
        # print(f"im: {im} | iax: {iax} | irow: {im // nCols} | icol: {im % nCols}")
        if "_notime" in classification_type:
            sns.scatterplot(x=np.linspace(-15, 15, len(clfs_included)),
                            y=metrics_included[iax], hue="model", data=df_metrics_avg, s=12,
                            ax=axs[im // nCols, im % nCols], palette=sns.color_palette(), edgecolor='black',
                            hue_order=clfs_included, legend=False)
        else:
            sns.lineplot(x="time", y=metrics_included[iax], hue="model", data=df_metrics_avg, legend=legend_switch,
                     ax=axs[im // nCols, im % nCols], hue_order=clfs_included, palette=sns.color_palette(), lw='0.7')

        axs[im // nCols, im % nCols].axhline(y=0.5, color='gray', ls='--', lw=0.5) if metrics_included[iax] in ['Accuracy', 'AUROC'] else []
        axs[im // nCols, im % nCols].axvline(x=0, color='gray', ls='--', lw=0.5)
        axs[im // nCols, im % nCols].set_title(metrics_included[iax], y=1, pad=-8)
        axs[im // nCols, im % nCols].set_xticks(xticks)
        axs[im // nCols, im % nCols].axes.xaxis.set_ticklabels([])
        axs[im // nCols, im % nCols].axes.yaxis.set_ticklabels([])
        axs[im // nCols, im % nCols].spines['top'].set_visible(False)
        axs[im // nCols, im % nCols].spines['right'].set_visible(False)
        axs[im // nCols, im % nCols].tick_params(direction="in")
        axs[im // nCols, im % nCols].set_xlim(xlim)
        axs[im // nCols, im % nCols].set_xlabel('')
        axs[im // nCols, im % nCols].set_ylabel('')
        if im // nCols == 0:
            axs[im // nCols, im % nCols].set_yticks(np.arange(0.4, 1.0, 0.1))
            axs[im // nCols, im % nCols].set_ylim([0.45, 1.0])
            axs[0, 0].set_yticklabels(['', '0.5', '0.6', '0.7', '0.8', '0.9'])
        if im // nCols == 1:
            axs[im // nCols, im % nCols].set_yticks(np.arange(0.1, 0.9, 0.1))
            axs[im // nCols, im % nCols].set_ylim([0.1, 0.9])
            axs[im // nCols, im % nCols].set_xticklabels(['', '0', '200', '400', '600'])
            axs[1, 0].set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8'])

    axs[1, 0].set_xlabel('Time / ms')
    axs[0, 1].legend(ncol=2, loc='center left', bbox_to_anchor=(1.1, 0.4), frameon=True)

    plt.show()


def plot_imetric_agestats(df_metrics, clfs_included, metrics_included, age_count, classification_type):

    np.random.seed(0)  # set seed for reproducibility in permutation statistics
    nCols = len(clfs_included)

    if "_notime" not in classification_type:
        times = np.unique(df_metrics["time"])
        df_metrics_avg = df_metrics.groupby(["time", "model", "age"], as_index=False)[metrics_included].mean()
        df_metrics_std = df_metrics.groupby(["time", "model", "age"], as_index=False)[metrics_included].std()

    fig, axs = plt.subplots(len(metrics_included), nCols, figsize=(9, 4))
    plt.subplots_adjust(hspace=0, wspace=0)  # top=0.88,bottom=0.11,left=0.08,right=0.97
    # clf_idx, clf_name = 0, clfs_included[0]
    for clf_idx, clf_name in enumerate(clfs_included):
        legend_switch = True if [clf_idx % nCols] == [0] else False
        if "_notime" not in classification_type:
            metric_avg_young_iclf = df_metrics_avg.loc[
                (df_metrics_avg["model"] == clf_name) & (df_metrics_avg["age"] == 0), metrics_included[0]]
            metric_std_young_iclf = df_metrics_std.loc[
                (df_metrics_std["model"] == clf_name) & (df_metrics_std["age"] == 0), metrics_included[0]]
            metric_avg_older_iclf = df_metrics_avg.loc[
                (df_metrics_avg["model"] == clf_name) & (df_metrics_avg["age"] == 1), metrics_included[0]]
            metric_std_older_iclf = df_metrics_std.loc[
                (df_metrics_std["model"] == clf_name) & (df_metrics_std["age"] == 1), metrics_included[0]]
            sns.lineplot(x="time", y=metrics_included[0], hue="age",
                         data=df_metrics_avg.loc[df_metrics_avg["model"] == clf_name, :], legend=legend_switch,
                         ax=axs[clf_idx % nCols], lw='0.7')
            axs[clf_idx % nCols].fill_between(times, metric_avg_young_iclf - metric_std_young_iclf,
                                              metric_avg_young_iclf + metric_std_young_iclf, alpha=0.3)
            axs[clf_idx % nCols].fill_between(times, metric_avg_older_iclf - metric_std_older_iclf,
                                              metric_avg_older_iclf + metric_std_older_iclf, alpha=0.3)
        else:
            sns.pointplot(x="age", y=metrics_included[0], hue="age",
                          data=df_metrics.loc[df_metrics["model"] == clf_name, :], dodge=5,
                          ax=axs[clf_idx % nCols])

        axs[clf_idx % nCols].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)
        axs[clf_idx % nCols].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
        axs[clf_idx % nCols].set_title(clf_name)
        axs[clf_idx % nCols].set_xticks(xticks)
        axs[clf_idx % nCols].set_xticklabels(['', '0', '200', '400', '600'])
        axs[clf_idx % nCols].set_yticks(np.arange(0.4, 1.0, 0.1))
        axs[clf_idx % nCols].axes.yaxis.set_ticklabels([])
        axs[clf_idx % nCols].set_xlim(xlim)
        axs[clf_idx % nCols].set_ylim([0.40, 0.95])
        axs[clf_idx % nCols].spines['top'].set_visible(False)
        axs[clf_idx % nCols].spines['right'].set_visible(False)
        axs[clf_idx % nCols].tick_params(direction="in")

        if clf_idx % nCols == 0:
            axs[clf_idx % nCols].legend(title='', frameon=True)  # prop={'size': fontsize-2})
            axs[clf_idx % nCols].set_ylabel('AUROC')
            axs[clf_idx % nCols].set_yticklabels(labels=['', '0.5', '0.6', '0.7', '0.8', '0.9'])
            axs[clf_idx % nCols].set_xlabel('Time / ms')
        else:
            axs[clf_idx % nCols].set_ylabel('')
            #axs[clf_idx % nCols].get_legend().remove()
            axs[clf_idx % nCols].set_xlabel('')

        # add statistics info to plot
        if "notime" not in classification_type:
            metric_young_iclf_forstats = np.array(
                df_metrics.loc[(df_metrics["model"] == clf_name) & (df_metrics["age"] == 0), metrics_included[0]],
                dtype=float).reshape(age_count[0], len(times))
            metric_older_iclf_forstats = np.array(
                df_metrics.loc[(df_metrics["model"] == clf_name) & (df_metrics["age"] == 1), metrics_included[0]],
                dtype=float).reshape(age_count[1], len(times))
            t_obs, clusters, cluster_p_values, h0 = permutation_cluster_test(
                [metric_young_iclf_forstats, metric_older_iclf_forstats],
                n_permutations=1024, tail=0, seed=1)
            for c, p_val in zip(clusters, cluster_p_values):
                if p_val <= 0.05:
                    print(c)
                    axs[clf_idx % nCols].scatter(times[c], np.ones((1, np.size(c))) * 0.91, s=0.5, c='black')

        # stats of notime version
        else:
            df_metrics2_iclf_young = df_metrics.loc[
                (df_metrics["model"] == clf_name) & (df_metrics["age"] == 0), metrics_included[0]]
            df_metrics2_iclf_older = df_metrics.loc[
                (df_metrics["model"] == clf_name) & (df_metrics["age"] == 1), metrics_included[0]]
            stat, p_value = ttest_ind(np.array(df_metrics2_iclf_young, dtype=float),
                                      np.array(df_metrics2_iclf_older, dtype=float))
            print(f"{clf_name} | auroc p-value: {p_value}")
            print(f"Mean time point of young | {clf_name} | {np.mean(df_metrics2_iclf_young)}")
            print(f"Mean time point of older | {clf_name} | {np.mean(df_metrics2_iclf_older)}")
            print(f"Std time point of young | {clf_name} | {np.std(df_metrics2_iclf_young)}")
            print(f"Std time point of older | {clf_name} | {np.std(df_metrics2_iclf_older)}")
            if p_value < 0.05:
                axs[clf_idx % nCols].scatter([20, 40], [0.86, 0.86], s=4, c='black', marker="*")

    plt.show()

def plot_feature_importance(df_importances, importance_metric_included, clfs_included, classification_type):

    feature_names = df_importances["feature"].cat.categories
    times = np.unique(df_importances["time"])

    df_importances = df_importances.dropna()
    df_importances_avg = df_importances.groupby(['time', 'model', 'feature'], as_index=False, sort=False)[
        importance_metric_included].mean()

    markers = ['o', 'o', 'x', 'd', 'd', 'd', '*', '*']
    colors = [sns.color_palette()[0]] * 4 + [sns.color_palette()[1]] * 4 + [sns.color_palette()[2]] * 4 + [
        sns.color_palette()[3]] * 4

    fig, axs = plt.subplots(1, len(clfs_included), figsize=(8, 6))
    plt.subplots_adjust(wspace=0, hspace=0)

    # clf_idx, clf_name = 0, clfs_included[0]
    for clf_idx, clf_name in enumerate(clfs_included):
        legend_switch = True if clf_idx == len(clfs_included)-1 else False
        if "notime" in classification_type:
            sns.scatterplot(x=np.random.uniform(low=-20, high=20, size=(len(feature_names),)),
                            y=importance_metric_included,
                            hue="feature",
                            style="feature",
                            s=20, data=df_importances_avg[df_importances_avg["model"] == clf_name],
                           # palette=sns.color_palette()[:len(feature_names)],
                            ax=axs[clf_idx],
                            edgecolor='black', legend=legend_switch, zorder=2)
        else:
            g = sns.lineplot(x="time", y=importance_metric_included,
                             hue="feature",
                          #   palette=sns.color_palette("pastel"),
                             data=df_importances_avg[df_importances_avg["model"] == clf_name],
                             legend=legend_switch,
                             lw='0.7',
                             ax=axs[clf_idx], zorder=1)

        axs[clf_idx].set_title(clfs_included[clf_idx])
        axs[clf_idx].set_ylim([-0.01, 0.30])
        axs[clf_idx].set_yticks(np.arange(0, 0.31, 0.10))
        axs[clf_idx].axes.yaxis.set_ticklabels([])
        axs[clf_idx].set_xlim(xlim)
        axs[clf_idx].set_xticks(xticks)
        axs[clf_idx].axes.xaxis.set_ticklabels(['', '0', '200', '400', '600'])
        axs[clf_idx].spines['top'].set_visible(False)
        axs[clf_idx].spines['right'].set_visible(False)
        axs[clf_idx].tick_params(axis='both', which='major', direction="in")
        axs[clf_idx].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        axs[clf_idx].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
        axs[clf_idx].set_xlabel("")
        axs[clf_idx].set_ylabel("")

        # if clf_idx == 2:
        #     axs[2].legend(title="", loc='center left', bbox_to_anchor=(1, 0.5))
        #     for t, l in zip(g.legend_.texts, feature_names):
        #         t.set_text(l)
        # else:
        #     axs[clf_idx].get_legend().remove()

    axs[0].set_yticklabels(['0', '0.1', '0.2', '0.3'])
    axs[0].set_xlabel('Time / ms')
    axs[0].set_ylabel("Feature importance")
    plt.show()

def plot_feature_importance_clfavg(df_importances, importance_metric_included, clfs_included, classification_type):

    feature_names = df_importances["feature"].cat.categories
    times = np.unique(df_importances["time"])

    df_importances = df_importances.dropna()
    df_importances_avg = df_importances.groupby(['time', 'feature'], as_index=False, sort=False)[
        importance_metric_included].mean()

    # markers = ['o', 'o', 'x', 'd', 'd', 'd', '*', '*']

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    plt.subplots_adjust(wspace=0, hspace=0)

    if "notime" in classification_type:
        sns.scatterplot(x=np.random.uniform(low=-20, high=20, size=(len(feature_names),)),
                        y=importance_metric_included,
                        hue="feature",
                        style="feature",
                        s=20, data=df_importances_avg,
                        #palette=sns.color_palette()[:len(feature_names)],
                        ax=axs,
                        edgecolor='black', zorder=2)
    else:
        g = sns.lineplot(x="time", y=importance_metric_included,
                         hue="feature",
                         #palette=sns.color_palette("pastel"),
                         data=df_importances_avg,
                         lw='0.7',
                         ax=axs, zorder=1)
    axs.set_ylim([-0.01, 0.30])
    axs.set_yticks(np.arange(0, 0.31, 0.10))
    axs.axes.yaxis.set_ticklabels([])
    axs.set_xlim(xlim)
    axs.set_xticks(xticks)
    axs.axes.xaxis.set_ticklabels(['', '0', '200', '400', '600'])
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.tick_params(axis='both', which='major', direction="in")
    axs.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    axs.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    axs.set_xlabel("")
    axs.set_ylabel("")
    axs.set_yticklabels(['0', '0.1', '0.2', '0.3'])
    axs.set_xlabel('Time / ms')
    axs.set_ylabel("Feature importance")
    plt.show()


def plot_cd_diagram_auroc(clf_dict, times, path_cd_true, path_cd_ranks):

    clfs_included = list(clf_dict.keys())

    average_ranks_overTime = pd.DataFrame(columns=clfs_included, index=times)
    p_values_RF_overTime = pd.DataFrame(columns=clfs_included, index=times)
    p_values_LDA_overTime = pd.DataFrame(columns=clfs_included, index=times)
    p_values_XGB_overTime = pd.DataFrame(columns=clfs_included, index=times)
    p_values_ADA_overTime = pd.DataFrame(columns=clfs_included, index=times)
    p_values_Log_overTime = pd.DataFrame(columns=clfs_included, index=times)
    p_values_KNN_overTime = pd.DataFrame(columns=clfs_included, index=times)
    p_values_SVCL_overTime = pd.DataFrame(columns=clfs_included, index=times)
    p_values_SVCRBF_overTime = pd.DataFrame(columns=clfs_included, index=times)
    p_values_Tree_overTime = pd.DataFrame(columns=clfs_included, index=times)

    p_values_df = [p_values_LDA_overTime, p_values_Log_overTime, p_values_SVCL_overTime, p_values_SVCRBF_overTime,
                   p_values_RF_overTime,
                   p_values_XGB_overTime, p_values_KNN_overTime, p_values_ADA_overTime, p_values_Tree_overTime]

    # idxclf, clf, it, t_val = 0, "LDA", 0, 730.46875
    for idxclf, clf in enumerate(clfs_included):
        for it, t_val in enumerate(times):
            with open(path_cd_ranks + 'cd_pvalues_' + str(it) + '.pkl', 'rb') as f:
                [p_values, average_ranks] = pickle.load(f)
                average_ranks_overTime.loc[t_val] = average_ranks

            try:
                with open(path_cd_true + 'cd_pvalues_' + str(it) + '.pkl', 'rb') as f:
                    [p_values, average_ranks] = pickle.load(f)
                    average_ranks_overTime.loc[t_val] = average_ranks
                arr = np.zeros((1, 9)).flatten()
                for il in p_values:
                    if (clf in il) and (il[3] is False):
                        print(il)
                        arr[[clf_dict.get(il[0]), clf_dict.get(il[1])]] = 1
                    else:
                        arr[clf_dict.get(clf)] = 1
                p_values_df[idxclf].loc[t_val] = arr
            except:
                p_values_df[idxclf].loc[t_val] = np.nan


    np_average_ranks_overTime = np.array(average_ranks_overTime.T, dtype=float)
    np_p_values_RF_overTime = np.where(p_values_RF_overTime.T == 1)
    np_p_values_XGB_overTime = np.where(p_values_XGB_overTime.T == 1)
    np_p_values_ADA_overTime = np.where(p_values_ADA_overTime.T == 1)
    np_p_values_Log_overTime = np.where(p_values_Log_overTime.T == 1)
    np_p_values_KNN_overTime = np.where(p_values_KNN_overTime.T == 1)
    np_p_values_SVCL_overTime = np.where(p_values_SVCL_overTime.T == 1)
    np_p_values_SVCRBF_overTime = np.where(p_values_SVCRBF_overTime.T == 1)
    np_p_values_LDA_overTime = np.where(p_values_LDA_overTime.T == 1)
    np_p_values_Tree_overTime = np.where(p_values_Tree_overTime.T == 1)

    plt.figure(figsize=(9, 4))
    plt.yticks(ticks=range(0, 9), labels=clfs_included)
    plt.imshow(np_average_ranks_overTime, cmap='gist_gray', aspect='auto', vmin=0, vmax=9)
    plt.plot(np_p_values_LDA_overTime[1], np_p_values_LDA_overTime[0] - 0.225, '*', c='tab:blue', markersize=3,label="LDA")
    plt.plot(np_p_values_RF_overTime[1], np_p_values_RF_overTime[0] - 0.075, '*', c='tab:purple', markersize=3,label="RF")
    # plt.plot(np_p_values_ADA_overTime[1], np_p_values_ADA_overTime[0] + 0.15, '*', c='tab:green', markersize=3,label="AdaBoost")
    # plt.plot(np_p_values_SVCRBF_overTime[1], np_p_values_SVCRBF_overTime[0]+0.05, 'c*', markersize=3)
    plt.plot(np_p_values_XGB_overTime[1], np_p_values_XGB_overTime[0]+0.075, '*', c='tab:green', markersize=3, label="XGB")
    # plt.plot(np_p_values_Log_overTime[1], np_p_values_Log_overTime[0]+0.1, 'c*', markersize=3)
    plt.plot(np_p_values_Tree_overTime[1], np_p_values_Tree_overTime[0]+0.225, '*', c='tab:red', markersize=3,label="Tree")
    # plt.plot(np_p_values_SVCL_overTime[1], np_p_values_SVCL_overTime[0], 'y+', markersize=5)
    # plt.plot(np_p_values_Tree_overTime[1], np_p_values_Tree_overTime[0]+0.2, 'k*', markersize=3)
    plt.axvline(x=31, color='black', linestyle='--', linewidth=0.5)
    plt.xticks(ticks=[31, 82, 133, 185], labels=['0', '200', '400', '600'])
    plt.xlabel('Time / ms')
    plt.legend(bbox_to_anchor=(1.0, 1.0), frameon=True)
    plt.colorbar(label='Ranks', shrink=0.7, anchor=(2, 0))
    plt.tick_params(direction="in")
    plt.show()


def plot_individual_max(df_metrics, clfs_included, metric_included):

    metric_max_vals = df_metrics.groupby(['sub', 'age', 'model'], as_index=False, sort=False).max().drop(['time', 'sub'], axis=1)
    min = np.min(metric_max_vals[metric_included])
    fig, axs = plt.subplots(1, len(clfs_included), squeeze=True)
    plt.subplots_adjust(wspace=0., hspace=0.)

    # clf_idx, clf_name = 0, clfs_included[0]
    for clf_idx, clf_name in enumerate(clfs_included):
        sns.violinplot(x="age", y=metric_included, hue_order=[0, 1], data=metric_max_vals[metric_max_vals["model"] == clf_name], color="0.8", ax=axs[clf_idx])
        sns.stripplot(x="age", y=metric_included, hue_order=[0, 1], data=metric_max_vals[metric_max_vals["model"] == clf_name], jitter=True, ax=axs[clf_idx])
        axs[clf_idx].set_ylim([min-0.1, 1.1])
        axs[clf_idx].set_title(clfs_included[clf_idx])
        axs[clf_idx].set_ylabel("")
        axs[clf_idx].set_xlabel("")
        axs[clf_idx].set_yticks(np.arange(0.5, 1.05, 0.1))
        axs[clf_idx].axes.xaxis.set_ticks([])
        axs[clf_idx].axes.yaxis.set_ticklabels([])
        axs[clf_idx].tick_params(direction="in")
        plt.setp(axs[clf_idx].collections, alpha=.5)

        #### Check statistical significance
        # print(f"Check statisticaly significant age differences in max amplitude")
        stat, p_value = ttest_ind(metric_max_vals.loc[(metric_max_vals["model"] == clf_name) & (metric_max_vals["age"]==0), metric_included].dropna(),
                                  metric_max_vals.loc[(metric_max_vals["model"] == clf_name) & (metric_max_vals["age"]==1), metric_included].dropna())
        print(f"{clf_name} | {metric_included} p-value: {p_value}")

        if p_value < 0.05:
            sns.scatterplot(x=[0.5], y=[1.055], marker='*', s=20, ax=axs[clf_idx], color='black')
            sns.lineplot(x=[0.1, 0.9], y=[1.05, 1.05], ax=axs[clf_idx], color='black')

    axs[0].set_yticklabels(['', '0.6', '0.7', '0.8', '0.9', '1.0'])
    axs[0].set_ylabel("AUROC")
    #axs[0].legend(labels=["", "", "young", "older"], title="", loc="best")
    plt.show()


def plot_timepoints_at_individual_max(df_metrics, clfs_included, metric_included):

    metric_idxmax = df_metrics.groupby(['sub', 'age', 'model'], as_index=False, sort=False).idxmax().dropna()
    time_of_max_metric = df_metrics.loc[metric_idxmax[metric_included], ['sub', 'age', 'model', 'time']]

    fig, axs = plt.subplots(1, len(clfs_included), squeeze=True)
    plt.subplots_adjust(wspace=0., hspace=0.)

    # clf_idx, clf_name = 0, clfs_included[0]
    for clf_idx, clf_name in enumerate(clfs_included):

        sns.violinplot(y="age", x="time", data=time_of_max_metric[time_of_max_metric["model"] == clf_name], color="0.8", orient='h', ax=axs[clf_idx], label=clf_name)
        sns.stripplot(y="age", x="time", data=time_of_max_metric[time_of_max_metric["model"] == clf_name], jitter=True, orient='h', ax=axs[clf_idx])
        axs[clf_idx].set_title(clfs_included[clf_idx])
        axs[clf_idx].set_ylabel("")
        axs[clf_idx].set_xlabel("")
        axs[clf_idx].set_xticks(xticks)
        axs[clf_idx].axes.yaxis.set_ticks([])
        axs[clf_idx].axes.yaxis.set_ticklabels([])
        axs[clf_idx].tick_params(direction="in")
        plt.setp(axs[clf_idx].collections, alpha=.5)

        #do stats
        time_of_max_metric_iclf_young = time_of_max_metric.loc[(time_of_max_metric["model"] == clf_name) & (time_of_max_metric["age"]==0), "time"]
        time_of_max_metric_iclf_older = time_of_max_metric.loc[(time_of_max_metric["model"] == clf_name) & (time_of_max_metric["age"]==1), "time"]
        stat, p_value = ttest_ind(time_of_max_metric_iclf_young, time_of_max_metric_iclf_older)
        if p_value < 0.05:
            sns.scatterplot(x=[400], y=[0.5], marker='*', s=20, ax=axs[clf_idx], color='black')
            sns.lineplot(x=[401, 401], y=[0.1, 0.9], ax=axs[clf_idx], color='black')

        print(f" {clf_name} | auroc p-value: {p_value}")
        print(f"Mean time point of young | {clf_name} | {np.mean(time_of_max_metric_iclf_young)}")
        print(f"Mean time point of older | {clf_name} | {np.mean(time_of_max_metric_iclf_older)}")
        print(f"Std time point of young | {clf_name} | {np.std(time_of_max_metric_iclf_young)}")
        print(f"Std time point of older | {clf_name} | {np.std(time_of_max_metric_iclf_older)}")

    axs[0].set_xlabel("Time / ms")

    plt.show()