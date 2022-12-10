import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
plt.interactive(False)
import scipy as sp
import numpy as np
from cycler import cycler
import scipy.io
from mne.stats import permutation_cluster_test
import pandas as pd
import pickle
from scipy.signal import savgol_filter
import seaborn as sns
from cd_diagram import draw_cd_diagram
from scipy.stats import ttest_ind
# from scikit_posthocs import posthoc_dunn

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
color_cycle = cycler(color=colors)
plt.rcParams['legend.labelcolor'] = 'black'
plt.rcParams['axes.prop_cycle'] = color_cycle

#################################################################################################################

def load_from_mat(filename=None, data={}, loaded=None):
    if filename:
        vrs = scipy.io.whosmat(filename)
        name = vrs[0][0]
        loaded = scipy.io.loadmat(filename, struct_as_record=True)
        loaded = loaded[name]
    whats_inside = loaded.dtype.fields
    fields = list(whats_inside.keys())
    for field in fields:
        if False and len(loaded[0, 0][field].dtype) > 0:  # it's a struct
            data[field] = {}
            data[field] = load_from_mat(data=data[field], loaded=loaded[0, 0][field])
        else:  # it's a variable
            data[field] = loaded[0, 0][field]
    return data

def calculate_cd_diagram(df_metrics, metrics_included, path_cd_ranks, path_cd_true):
    os.makedirs(path_cd_ranks, exist_ok=True)
    os.makedirs(path_cd_true, exist_ok=True)
    times = np.unique(df_metrics["time"])
    df_metrics_reduced = df_metrics[["time", "sub", "model", metrics_included]]
    df_metrics_reduced = df_metrics_reduced.rename(columns = {"sub": "dataset_name", "model": "classifier_name"})

    # it, t_val = 0, times[0]
    for it, t_val in enumerate(times):
        df_metrics_reduced_it = df_metrics_reduced.loc[df_metrics['time'] == t_val].drop("time", axis=1)
        draw_cd_diagram(path_cd_true, it, df_perf=df_metrics_reduced_it, title=metrics_included, labels=True, metric=metrics_included, noposthoc=False, path_cd_ranks=path_cd_ranks)


# plot features averaged over trials and grouped by classes
def plot_features(data, times, labels, label_names, feature_names, classification_type, fig_path=None, data_version=None):

    nFeatures = len(feature_names)

    if "notime" in classification_type:
        feature_names_edited = [feature_names[i][0][2:].replace('_', ': ') for i in range(len(feature_names))]
        feature_names_edited[2] = 'N170: PA'
        feature_names_edited[3] = 'N170: FL'
        feature_order = [0, 2, 4, 6, 1, 3, 5, 7]

        row_ylims = [[-51, 61]]*4 + [[-10, 510]]*4
        row_yticks = [np.arange(-40, 45, 20)]*4 + [np.arange(0, 520, 100)]*4
        row_yticklabels = [['-40', '-20', '0', '20', '40'], [], [], [],
                           ['0', '', '200', '', '400', ''], [], [], []]
        vp_facecolors = ['tab:blue', 'tab:orange', 'lightblue', 'goldenrod']

        fig, axs = plt.subplots(2, 4, figsize=(7, 4))
        plt.subplots_adjust(wspace=0, hspace=0.1)

        for i, ax in enumerate(fig.axes):
            # print(f"i: {i} | feature_order: {feature_order[i]} | feature: {feature_names[feature_order[i]]}")
            data_grouped = [data[labels==3, feature_order[i]],
                            data[labels==1, feature_order[i]],
                            data[labels==2, feature_order[i]],
                            data[labels==0, feature_order[i]]]
            vp = ax.violinplot(data_grouped, positions=[1, 2, 3, 4], showmeans=False, showextrema=False)
            for ip, vph in enumerate(vp['bodies']):
                vph.set_facecolor(vp_facecolors[ip])
                vph.set_edgecolor(vp_facecolors[ip])
                vph.set_alpha(0.7)

            quartile1, medians, quartile3 = np.zeros(4), np.zeros(4), np.zeros(4)
            for ip in range(4):
                quartile1[ip], medians[ip], quartile3[ip] = np.percentile(data_grouped[ip], [25, 50, 75], axis=0)
            inds = np.arange(1, len(medians) + 1)
            ax.scatter(inds, medians, marker='o', color='k', s=3, zorder=3)
            ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=1)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            ax.set_ylim(row_ylims[i])
            ax.set_yticks(row_yticks[i])
            ax.set_yticklabels(row_yticklabels[i]) if i in [0, 4] else ax.set_yticklabels('')
            ax.set_xticks([])
            ax.set_title(feature_names_edited[feature_order[i]], pad=-10)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='in')

        axs[1, 0].legend([label_names[3][0], label_names[1][0], label_names[2][0], label_names[0][0]], loc="upper right", title="", ncol=1, prop={'size': 5})
        axs[0, 0].set_ylabel('Amplitude / \u03BCV')
        axs[1, 0].set_ylabel('Latency / ms')
        plt.show()
    else:
        feature_names = [feature_names[i][0][2:] for i in range(len(feature_names))]
        # feature_names = ["Occipital amplitude", "Parietal amplitude", "Central amplitude", "Frontal amplitude",
        #                  "Occipital 32 Hz power", "Parietal 32 Hz power", "Central 8 Hz power", "Central 32 Hz power"]
        # feature_names = ["Occipital amplitude", "Central amplitude", "Occipital 4 Hz power",
        #                  "Occipital 32 Hz power", "Parietal 4 Hz power", "Parietal 32 Hz power", "Central 32 Hz power", "Frontal 12 Hz power"]
        feature_order_map = {0: 0, 1: 2, 2: 3, 3: 7, 4: 1, 5: 4, 6: 5, 7: 6}
        feature_names_new = ["Occipital", "Occipital 4 Hz", "Occipital 8 Hz", "Frontal 4 Hz",
                         "Central", "Central 4 Hz", "Central 8 Hz", "Central 12 Hz"]
        fig, axs = plt.subplots(2, 4, figsize=(7, 4))
        plt.subplots_adjust(wspace=0., hspace=0.)

        for i, ax in enumerate(fig.axes):
            #print(i, feature_names[feature_order_map[i]], feature_names_new[i])
            ax.plot(times, np.mean(data[labels == 3, feature_order_map[i], :], 0), label=label_names[3][0], color='tab:blue', linestyle='-')
            ax.plot(times, np.mean(data[labels == 1, feature_order_map[i], :], 0), label=label_names[1][0], color='tab:orange', linestyle='-')
            ax.plot(times, np.mean(data[labels == 2, feature_order_map[i], :], 0), label=label_names[2][0], color='lightblue', linestyle='--')
            ax.plot(times, np.mean(data[labels == 0, feature_order_map[i], :], 0), label=label_names[0][0], color='goldenrod', linestyle='--')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
            ax.set_title(feature_names_new[i], y=1, pad=-10)
            ax.set_xticks(xticks)
            ax.set_yticks(np.insert(np.arange(-4, 6, 4), 3, 7))
            ax.set_yticklabels(['', '', '', ''])
            ax.set_ylim([-6, 7])
            ax.set_xlim(xlim)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(direction='in')
            ax.set_xticklabels(['', '0', '200', '400', '600']) if i > 3 else ax.set_xticklabels(['', '', '', '', ''])

        axs[1, 0].set_xlabel('Time / ms')
        axs[0, 0].set_ylabel('Amplitude / \u03BCV')
        axs[1, 0].set_ylabel('Amplitude / \u03BCV')
        #axs[0, 1].set_ylabel('ERSP / dB')
        #axs[1, 1].set_ylabel('ERSP / dB')
        axs[0, 0].set_yticklabels(['-4', '0', '4', ''])
        #axs[0, 1].set_yticklabels(['-4', '0', '4', ''])
        axs[1, 0].set_yticklabels(['-4', '0', '4', ''])
        #axs[1, 1].set_yticklabels(['-4', '0', '4', ''])
        axs[1, 3].legend(bbox_to_anchor=(0.6, -0.1), ncol=4, loc='best',  frameon=True, prop={'size': 6})

    plt.show()
    plt.savefig(f"{fig_path}\\plot_features_dat{data_version}_plotv5.png", bbox_inches='tight')
    plt.close('all')

def plot_features_by_label(times, data, labels, feature_names):
    nFeatures = data.shape[1]
    nRows = 3
    nCols = nFeatures // nRows

    fig, axs = plt.subplots(nRows, nCols, figsize=(4, 9))
    plt.subplots_adjust(wspace=0, hspace=0.01)
    map_dict = {0:1, 1:0, 2:4, 3:2, 4:5, 5:3}

    for i, ax in enumerate(fig.axes):
        # print(f"i: {i} | maped i: {map_dict[i]} | feature: {feature_names[map_dict[i]]}")
        ax.plot(times, np.mean(data[labels == 4, map_dict[i], :], 0), label='rare-young', color='tab:blue', linestyle='-')
        ax.plot(times, np.mean(data[labels == 2, map_dict[i], :], 0), label='rare-older', color='tab:orange', linestyle='-')
        ax.plot(times, np.mean(data[labels == 3, map_dict[i], :], 0), label='freq-young', color='lightblue', linestyle='--')
        ax.plot(times, np.mean(data[labels == 1, map_dict[i], :], 0), label='freq-older', color='goldenrod', linestyle='--')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xticks(xticks)
        ax.set_yticks(np.arange(-4, 6, 4))
        ax.set_yticklabels(['', '', ''])
        ax.set_xticklabels(['', '', '', '', ''])
        ax.set_ylim([-6, 7])
        ax.set_xlim(xlim)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='in')

    axs[0, 0].set_title('CENTRAL', y=1.0, pad=0)
    axs[0, 1].set_title('OCCIPITAL', y=1.0, pad=0)
    axs[1, 0].set_title('8 Hz', y=1.0, pad=-12)
    axs[1, 1].set_title('8 Hz', y=1.0, pad=-12)
    axs[2, 0].set_title('16 Hz', y=1.0, pad=-12)
    axs[2, 1].set_title('16 Hz', y=1.0, pad=-12)
    axs[2, 0].set_xlabel('Time / ms')
    axs[0, 0].set_ylabel('Amplitude / \u03BCV')
    axs[1, 0].set_ylabel('ERSP / dB')
    axs[2, 0].set_ylabel('ERSP / dB')
    axs[2, 0].set_xticklabels(['', '0', '200', '400', '600'])
    axs[2, 1].set_xticklabels(['', '0', '200', '400', '600'])
    axs[0, 0].set_yticklabels(['-4', '0', '4'])
    axs[1, 0].set_yticklabels(['-4', '0', '4'])
    axs[2, 0].set_yticklabels(['-4', '0', '4'])
    axs[2, 0].legend(loc='lower center', ncol=2, frameon=True, prop={'size': 5})

    plt.show()

# plot features averaged over trials and grouped by classes
def plot_features_by_label_notime(age_coded, data, labels, feature_names):

    nFeatures = data.shape[1]
    nRows = 4
    nCols = nFeatures // nRows
    # delete 2 trials that are complete outliers
    delidxManual = [422, 1397, 1651, 1680, 1681, 3629, 3630, 3872, 4112, 4113, 4260, 4265, 5484, 5497, 5725, 5726, 5727, 6019, 6313]
    delidx1 = np.where((data[:, [0, 4, 8, 12]] > 50) | (data[:, [0, 4, 8, 12]] < -20))
    delidx2 = np.where((data[:, [1, 5, 9, 13]] > 20) | (data[:, [1, 5, 9, 13]] < -50))
    delidx3 = np.where((data[:, [2, 6, 10, 14]] < 0) | (data[:, [2, 6, 10, 14]] > 500))
    delidx4 = np.where((data[:, [3, 7, 11, 15]] < 0) | (data[:, [3, 7, 11, 15]] > 500))
    delall = np.unique(np.concatenate((delidx1[0], delidx2[0], delidx3[0], delidx4[0], delidxManual)))
    dataB = np.delete(data, delall, axis=0)
    age_codedB = np.delete(age_coded, delall, axis=0)
    labelsB = np.delete(labels, delall, axis=0)
    # some plot settings
    col_names = ['P1', 'N1', 'P2', 'P3']
    row_names = ['PA', 'MA', 'PL', 'FL']
    row_ylims = [[-60, 60], [-60, 60], [-10, 520], [-10, 520]]
    row_yticks = [np.arange(-40, 45, 20), np.arange(-40, 45, 20), np.arange(0, 520, 100), np.arange(0, 520, 100)]
    row_yticklabels = [['-40',  '-20',   '0',   '20',  '40'], ['-40',  '-20',   '0',   '20',  '40'], [  '0', '', '200', '', '400', ''], [  '0', '', '200', '', '400', '']]
    vp_facecolors = ['tab:blue', 'tab:orange', 'lightblue', 'goldenrod']
    legendlabels = ['rare-young', 'rare-older', 'freq-young', 'freq-older']
    feature_order = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]

    fig, axs = plt.subplots(nRows, nCols, figsize=(4, 9))
    plt.subplots_adjust(wspace=0, hspace=0.01)

    for i, ax in enumerate(fig.axes):
        #print(f"i: {i} | feature_order: {feature_order[i]} | feature: {feature_names[feature_order[i]]}")
        dataC = [dataB[(age_codedB == 0) & (labelsB == 1), feature_order[i]],
                 dataB[(age_codedB == 1) & (labelsB == 1), feature_order[i]],
                 dataB[(age_codedB == 0) & (labelsB == 0), feature_order[i]],
                 dataB[(age_codedB == 1) & (labelsB == 0), feature_order[i]]]
        vp = ax.violinplot(dataC, positions=[1, 2, 3, 4], showmeans=False, showextrema=False)
        for ip, vph in enumerate(vp['bodies']):
            vph.set_facecolor(vp_facecolors[ip])
            vph.set_edgecolor(vp_facecolors[ip])
            vph.set_alpha(0.7)

        quartile1, medians, quartile3 = np.zeros(4), np.zeros(4), np.zeros(4)
        for ip in range(4):
            quartile1[ip], medians[ip], quartile3[ip] = np.percentile(dataC[ip], [25, 50, 75], axis=0)
        inds = np.arange(1, len(medians) + 1)
        ax.scatter(inds, medians, marker='o', color='k', s=3, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=1)

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_ylim(row_ylims[i // 4])
        ax.set_yticks(row_yticks[i // 4])
        ax.set_xticks([])
        ax.set_yticklabels(row_yticklabels[i // 4]) if i % 4 == 0 else ax.set_yticklabels('')
        ax.set_title(col_names[i % 4]) if i // 4 == 0 else ''
        ax.set_ylabel(row_names[i // 4]) if i % 4 == 0 else ''
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='in')
        if i == 13:
            ax.legend(legendlabels, title="", ncol=2, prop={'size': 5})

    plt.show()

def stats(dataB, age_codedB, labelsB, feature_names, legendlabels):

    for i, ifeature in enumerate(feature_names):
        #i, ifeature = 2, feature_names[2]
        dataC = [dataB[(age_codedB == 0) & (labelsB == 1), i],
                 dataB[(age_codedB == 1) & (labelsB == 1), i],
                 dataB[(age_codedB == 0) & (labelsB == 0), i],
                 dataB[(age_codedB == 1) & (labelsB == 0), i]]

        s, p = sp.stats.kruskal(dataC[0], dataC[1], dataC[2], dataC[3])
        if p < 0.05:
            p_post = posthoc_dunn(dataC, p_adjust='fdr_bh')
            print(f'Feature name: {ifeature}. KW was significant. Doing post hoc. '
                  f'P-value matrix is:')
            print(p_post)
            print(f"Group order:  {legendlabels}")
        else:
            print(f'Feature name: {ifeature}. KW test was not significant. No posthoc.')

# plot features averaged over trials and grouped by classes, extended for 4 electrode clusters
def plot_features_by_label_extended(times, data, labels, feature_names):
    nFeatures = data.shape[1]
    nCols = 4
    nRows = nFeatures // nCols

    fig, axs = plt.subplots(nRows, nCols, figsize=(4, 9))
    plt.subplots_adjust(wspace=0, hspace=0.01)
    #map_dict = {0:3, 0:1, 1:0, 2:4, 3:2, 4:5, 5:3}

    for i, ax in enumerate(fig.axes):
        print(f"i: {i} | feature: {feature_names[i]} ") #| maped i: {map_dict[i]} | feature: {feature_names[map_dict[i]]}")
        ax.plot(times, np.mean(data[labels == 4, i, :], 0), label='rare-young', color='tab:blue', linestyle='-')
        ax.plot(times, np.mean(data[labels == 2, i, :], 0), label='rare-older', color='tab:orange', linestyle='-')
        ax.plot(times, np.mean(data[labels == 3, i, :], 0), label='freq-young', color='lightblue', linestyle='--')
        ax.plot(times, np.mean(data[labels == 1, i, :], 0), label='freq-older', color='goldenrod', linestyle='--')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xticks(xticks)
        ax.set_yticks(np.arange(-4, 6, 4))
        ax.set_yticklabels(['', '', ''])
        ax.set_xticklabels(['', '', '', '', ''])
        ax.set_ylim([-6, 7])
        ax.set_xlim(xlim)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='in')

    axs[0, 0].set_title('CENTRAL', y=1.0, pad=0)
    axs[0, 1].set_title('OCCIPITAL', y=1.0, pad=0)
    axs[1, 0].set_title('8 Hz', y=1.0, pad=-12)
    axs[1, 1].set_title('8 Hz', y=1.0, pad=-12)
    axs[2, 0].set_title('16 Hz', y=1.0, pad=-12)
    axs[2, 1].set_title('16 Hz', y=1.0, pad=-12)
    axs[2, 0].set_xlabel('Time / ms')
    axs[0, 0].set_ylabel('Amplitude / \u03BCV')
    axs[1, 0].set_ylabel('ERSP / dB')
    axs[2, 0].set_ylabel('ERSP / dB')
    axs[2, 0].set_xticklabels(['', '0', '200', '400', '600'])
    axs[2, 1].set_xticklabels(['', '0', '200', '400', '600'])
    axs[0, 0].set_yticklabels(['-4', '0', '4'])
    axs[1, 0].set_yticklabels(['-4', '0', '4'])
    axs[2, 0].set_yticklabels(['-4', '0', '4'])
    axs[2, 0].legend(loc='lower center', ncol=2, frameon=True, prop={'size': 5})

    plt.show()


def plot_feature_ranking_notime(df_importances, importance_metrics_included, clfs_included, feature_names, feature_names_nice, classification_type, time_switch):

    if classification_type == "bysub" + time_switch:
        markers = ['o', 'x', 'd', '*'] *4
        colors = [sns.color_palette()[0]] * 4 + [sns.color_palette()[1]] * 4 + [sns.color_palette()[2]] * 4 + [sns.color_palette()[3]] * 4
        fig, axs = plt.subplots(1, len(clfs_included), figsize=(9, 4))
        plt.subplots_adjust(hspace=0,wspace=0.20) #top=0.88,bottom=0.11,left=0.08,right=0.97
        # clf_idx, clf_name = 0, clfs_included[0]
        for clf_idx, clf_name in enumerate(clfs_included):

            sns.pointplot(x="age", y=importance_metrics_included, hue="feature", ax=axs[clf_idx], dodge=True, scale=0.5, errwidth=0.5,
                          data=df_importances[df_importances["model"] == clf_name], markers=markers, palette=colors)
            axs[clf_idx].set_title(clf_name)
            axs[clf_idx].legend([]) if clf_idx != 1 else axs[clf_idx].legend(ncol=8, frameon=True, loc="upper center", bbox_to_anchor=(0.9, -0.05))
            axs[clf_idx].spines['top'].set_visible(False)
            axs[clf_idx].spines['right'].set_visible(False)
            axs[clf_idx].axes.xaxis.set_ticklabels(["young", "older"])
            axs[clf_idx].set_xlabel("")
            axs[clf_idx].set_ylabel("")
            axs[clf_idx].tick_params(axis='both', which='major', direction="in")

        axs[0].set_ylabel("Feature importance")
        #axs[0].set_xlabel("Age group")
        plt.show()
    else:
        markers = ['o', 'x', 'd', '*'] * 4
        colors = [sns.color_palette()[0]] * 4 + [sns.color_palette()[1]] * 4 + [sns.color_palette()[2]] * 4 + [
            sns.color_palette()[3]] * 4
        fig, axs = plt.subplots(1, len(clfs_included), figsize=(9, 4))
        plt.subplots_adjust(hspace=0, wspace=0.20)  # top=0.88,bottom=0.11,left=0.08,right=0.97
        # clf_idx, clf_name = 0, clfs_included[0]
        for clf_idx, clf_name in enumerate(clfs_included):
            sns.pointplot(x="feature", y=importance_metrics_included, hue="feature", ax=axs[clf_idx], dodge=True, scale=0.5,
                          errwidth=0.5,
                          data=df_importances[df_importances["model"] == clf_name], markers=markers, palette=colors)
            axs[clf_idx].set_title(clf_name)
            axs[clf_idx].legend([]) if clf_idx != 1 else axs[clf_idx].legend(ncol=5, frameon=True, loc="upper center",
                                                                             bbox_to_anchor=(0.9, -0.05))
            axs[clf_idx].spines['top'].set_visible(False)
            axs[clf_idx].spines['right'].set_visible(False)
            axs[clf_idx].set_xlabel("")
            axs[clf_idx].set_ylabel("")
            axs[clf_idx].tick_params(axis='both', which='major', direction="in")

        axs[0].set_ylabel("Feature importance")
        axs[0].set_xlabel("Age group")
        plt.show()

def plot_feature_ranks_notime(df_ranking, metric):

    df_ranking["feature"].cat.reorder_categories(feature_names, inplace=True)
    df_ranking["feature"].cat.rename_categories([df_ranking["feature"].cat.categories[i][2:].replace('_', ': ') for i in range(len(df_ranking["feature"].cat.categories))], inplace=True)
    df_ranking_avg = df_ranking.groupby(["feature"], as_index=False)[metric].mean()

    fig, ax = plt.subplots()
    h1 = sns.pointplot(x="feature", y=metric, data=df_ranking, hue="model", palette=sns.color_palette("pastel"),
                        scale=0.5, errwidth=0.5, ax=ax)
    plt.setp(h1.lines, zorder=100)
    plt.setp(h1.collections, zorder=100)
    h2 = sns.pointplot(x="feature", y=metric, data=df_ranking_avg, color="black", scale=0.7, ax=ax, label="Average")
    plt.setp(h2.lines, zorder=1)
    plt.setp(h2.collections, zorder=1)
    # plt.axhline(y=3.5, color='gray', linestyle='--', linewidth=1)
    ax.axes.xaxis.set_ticklabels(df_ranking["feature"].unique(), rotation= 45)
    ax.legend(title="")
    plt.xlabel("Features")
    ylabel = "Rank" if metric == "ranking" else "Mutual information"
    plt.ylabel(ylabel)
    plt.ylim([0, 4])
    plt.show()

    #####################################3333
    # df_ranking_avg["mi"] = df_ranking_avg["mi"]/10
    # plt.figure(figsize=(2, 4))
    # h = sns.scatterplot(x=np.arange(-0.4, 0.4, 0.05), y=metric, data=df_ranking_avg, hue="feature", style="feature",
    #                     markers=['s']*4+['*']*4+['d']*4+['o']*4)
    # plt.ylabel("Mutual information")
    # plt.ylim([0, 0.5])
    # plt.yticks(np.arange(0, 0.5, 0.1))
    # plt.xticks([])
    # plt.xlabel('')
    # plt.legend(title="", ncol=2)
    # plt.show()


def plot_feature_ranks(df_ranking, metric, feature_names):

    feature_names_edited = ["O-amp", "P-amp", "C-amp", "F-amp"] + [i[4]+'-'+i[5:]+'Hz' for i in feature_names[4:]]
    df_ranking["feature"] = df_ranking["feature"].cat.reorder_categories(feature_names)
    df_ranking["feature"] = df_ranking["feature"].cat.rename_categories(feature_names_edited)
    df_ranking["mi"] = df_ranking["mi"] / 10         # correction because of 10 folds
    df_ranking_avg = df_ranking.groupby(["feature"], as_index=False)[metric].mean()

    fig, ax = plt.subplots()
    sns.pointplot(x="feature", y=metric, data=df_ranking_avg, palette=sns.color_palette(),
                       scale=0.5, errwidth=0.5, ax=ax)
    #plt.axhline(y=3.5, color='gray', linestyle='--', linewidth=1)
    ax.axes.xaxis.set_ticklabels(df_ranking["feature"].unique(), rotation=45)
    ax.axes.xaxis.set_ticklabels([])
    ax.legend(title="", ncol=4)
    plt.grid(True, alpha=0.5)
    plt.xlabel("Features")
    plt.ylabel("Mutual information")
    plt.show()

    colors = sns.color_palette()[:4] + sns.color_palette() + sns.color_palette() + sns.color_palette() + sns.color_palette()
    markers =['s']*4+['v']*9+['P']*9+['o']*9+['d']*9

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.scatterplot(x=np.linspace(-0.5, 0.5, nFeatures), y=metric, data=df_ranking_avg, hue="feature", style="feature",
                         markers=markers, palette=colors)
    plt.axhline(y=0.0820, color='gray', linestyle='--', linewidth=0.5)
    ax.axes.xaxis.set_ticklabels([])
    plt.ylim([0, 0.11])
    plt.yticks(np.arange(0, 0.11, 0.02))
    plt.xticks([])
    ax.legend(title="", ncol=5, loc="center", bbox_to_anchor=(0.5, 0.2), prop={'size': 5})
    plt.xlabel("Features")
    plt.ylabel("Mutual information")
    plt.show()




