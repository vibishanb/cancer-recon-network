"""
<!---------------------------
Name: cancer-recon-network
File: figure-plan
-----------------------------
Author: bvibishan
Data:   30/10/2025, 14:07:09
---------------------------->
"""

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection

import pickle
import os
from tqdm import tqdm

SMALL_SIZE = 15
MEDIUM_SIZE = 17
BIGGER_SIZE = 19

plt.rc('font', size=SMALL_SIZE, family='sans-serif', serif='Arial')          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('text')

def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

def calculate_overlap_stats(x, n_ct, max_links, met_ID):

    x_consumption = x[:max_links]
    x_production = x[max_links:]

    ### Consumption overlap
    con_index_overlap, con_mets_overlap = [[]], []
    for i, arr in enumerate(x_consumption.reshape(n_ct, -1).T):
        indices = np.where(arr == 1)[0]
        if len(indices) > 1:
            con_index_overlap.append(indices+1)
            con_mets_overlap.append(met_ID[i])
    
    ### Production overlap
    pro_index_overlap, pro_mets_overlap = [[]], []
    for i, arr in enumerate(x_production.reshape(n_ct, -1).T):
        indices = np.where(arr == 1)[0]
        if len(indices) > 1:
            pro_index_overlap.append(indices+1)
            pro_mets_overlap.append(met_ID[i])
    
    clist = ["+".join(map(str, ind.tolist())) for ind in con_index_overlap[1:]]
    plist = ["+".join(map(str, ind.tolist())) for ind in pro_index_overlap[1:]]

    return [clist, con_mets_overlap, plist, pro_mets_overlap]


home_dir = os.getcwd()
slope_df, sublinear_cell_lines = pd.read_pickle(home_dir + '/cancer_power_law_stats.pickle')

celltype_ID, celltypefreq, ec_metabolome_ID, ec_metabolome, met_baseline, core_mean = pd.read_pickle(home_dir + '/1-cells-data.pickle')
i_sublinear = np.where(np.isin(ec_metabolome.columns.values, sublinear_cell_lines))[0]
# ec_metabolome = ec_metabolome.iloc[:, i_sublinear]
diet = met_baseline.mean(axis=1)
ec_metabolome = ec_metabolome.iloc[:, i_sublinear]

"""
Figure 1 has been generated using a self-contained script called cancer-power-law.py that includes all the data filtering, calculations and plotting.
"""
# %%
#### Figure 2: Balanced production in one vs two celltypes, without learning
#### Data import
# cl='A549-ATCC'
# n_ct = 2
for n_ct in tqdm(np.arange(2, 8), desc='N_CT: '):
    for cl in tqdm(sublinear_cell_lines, desc='Cell line: '):
        pickle_path = '../raw-output/'+str(n_ct)+'-celltypes/no-learn-balanced-net/'+cl


        [balance_arr_1ct, balance_arr_nct, balance_flag_arr, balanced_networks_list,
                            rmse_arr_1ct, rmse_arr_nct,
                            ec_pred_arr_1ct, ec_pred_arr_nct,
                            index_arr_1ct, index_arr_nct,
                            production_ct_arr_1ct, production_ct_arr_nct] = pd.read_pickle(pickle_path + '/balanced-networks.pickle')
        n_reps = len(balance_flag_arr)
        n_rand = 100
        npro = 115
        ec_real = ec_metabolome.loc[:, cl].values

        net_state = 'no-learn-balanced-net/'
        figsave_flag = 1
        fig_path = '../figures/'+str(n_ct)+'-celltypes/'+net_state+cl
        try:
            os.makedirs(fig_path)
        except:
            pass

        df = pd.DataFrame({'Replicate': np.concatenate([np.arange(1, n_rand+1), np.arange(1, n_rand+1)]),
                        'N_CT': np.array([np.ones_like(rmse_arr_1ct), np.zeros_like(rmse_arr_1ct)+n_ct]).ravel().astype(int),
                        'Balance': np.concatenate([balance_flag_arr, balance_flag_arr]),
                        'X_pro': np.concatenate([balance_arr_1ct, balance_arr_nct[:, 1]]),
                        'RMSE': np.concatenate([rmse_arr_1ct, rmse_arr_nct])})

        g = sns.lineplot(data=df, x='N_CT', y='RMSE',
                        units='Replicate', hue='Balance',
                        palette='coolwarm_r', hue_norm=(0, 1),
                        dashes=False, estimator=None, zorder=1, legend=True)
        g = sns.scatterplot(data=df, x='N_CT', y='RMSE',
                            hue='Balance', palette='coolwarm_r', hue_norm=(0, 1),
                            edgecolor='face', alpha=0.8, zorder=2, legend=False)
        g.set_xlabel(r'$N_{CT}$')
        g.set(xlim=(0.5, n_ct+0.5), xticks=[1, n_ct])

        if figsave_flag:
            g.figure.savefig(fig_path+'/balance-pairwise-comparison-npro-'+str(npro)+'.png', dpi=300, bbox_inches='tight')
            plt.close(g.figure)
        else:
            plt.show()

        h = sns.boxplot(data=df, x='Balance', y='RMSE',
                        hue='N_CT', palette='crest')
        h.set_xlabel('Balance state')
        h.set(xlim=(-0.75, 1.75), xticks=[0, 1], xticklabels=['False', 'True'])
        h.get_legend().set_title(r'$N_{CT}$')
        if figsave_flag:
            h.figure.savefig(fig_path+'/balance-comparison-boxplot-npro-'+str(npro)+'.png', dpi=300, bbox_inches='tight')
            plt.close(h.figure)
        else:
            plt.show()


        if n_ct == 2:
            prod_df = pd.DataFrame({'Xpro_CT1': balance_arr_nct[:, 0],
                                    'Xpro_CT2': balance_arr_nct[:, 1],
                                    'RMSE': rmse_arr_nct})
            q = sns.scatterplot(data=prod_df, x='Xpro_CT1', y='Xpro_CT2',
                                hue = 'RMSE', palette='flare',#sizes=(50, 200),
                        # hue='N_produced', palette='crest',
                        edgecolor= 'face', alpha = 0.8, s=75)
            q.set_xscale('log')
            q.set_yscale('log')
            q.set_xlabel(r'$\chi_{production, CT1}$')
            q.set_ylabel(r'$\chi_{production, CT2}$')
            q.axvspan(xmin=0.1, xmax=1, alpha=0.2, color='tab:green')
            q.axhspan(ymin=0.1, ymax=1, alpha=0.2, color='tab:green')
            # q.axhline(y=1, linestyle='dashed', c='tab:green')
            # q.text(0.9, 0.38, r'$y=1$', c='tab:green', transform=g.transAxes)
            # q.axvline(x=1, linestyle='dashed', c='tab:green')
            # q.text(0.68, 0.87, r'$x=1$', c='tab:green', rotation='vertical', transform=g.transAxes)

        else:
            prod_df = pd.DataFrame(balance_arr_nct, columns=np.arange(1, n_ct+1)).join(pd.Series(rmse_arr_nct, name='RMSE')).melt(id_vars='RMSE', value_name='Balance', var_name='N_CT')

            q = sns.scatterplot(data=prod_df, x='Balance', y='RMSE',
                                hue = 'N_CT', palette='flare',#sizes=(50, 200),
                        # hue='N_produced', palette='crest',
                        edgecolor= 'face', alpha = 0.8, s=75)
            q.set_xscale('log')
            q.axvspan(xmin=0.1, xmax=1, alpha=0.2, color='tab:green')
            # q.set_yscale('log')
            q.set_xlabel(r'$\chi_{production}$')
            q.set_ylabel('RMSE')


        if figsave_flag:
            q.figure.savefig(fig_path+'/balance-and-rmse-lineplot-npro-'+str(npro)+'.png', dpi=300, bbox_inches='tight')
            plt.close(q.figure)
        else:
            plt.show()


        f, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8.5, 3.5))

        rmse_diff = rmse_arr_1ct - rmse_arr_nct
        i_min_diff = np.where(rmse_diff == rmse_diff.min(), True, False)
        i_max_diff = np.where(rmse_diff == rmse_diff.max(), True, False)

        #### Minimum difference
        colors = np.where(production_ct_arr_1ct[i_min_diff]==1, 'b', 
                        np.where(production_ct_arr_1ct[i_min_diff]==2, 'g',
                                np.where(production_ct_arr_1ct[i_min_diff]==3, 'darkorchid',
                                            np.where(production_ct_arr_1ct[i_min_diff]==4, 'saddlebrown',
                                                    np.where(production_ct_arr_1ct[i_min_diff]==5, 'goldenrod',
                                                             np.where(production_ct_arr_1ct[i_min_diff]==6, 'black', 'tab:pink'))))))
        labels = np.where(production_ct_arr_1ct[i_min_diff]==1, 'CT1', 
                        np.where(production_ct_arr_1ct[i_min_diff]==2, 'CT2',
                                np.where(production_ct_arr_1ct[i_min_diff]==3, 'CT3',
                                            np.where(production_ct_arr_1ct[i_min_diff]==4, 'CT4',
                                                    np.where(production_ct_arr_1ct[i_min_diff]==5, 'CT5',
                                                             np.where(production_ct_arr_1ct[i_min_diff]==6, 'CT6', 'CT7'))))))

        ax[0].scatter(np.log10(ec_pred_arr_1ct[i_min_diff][index_arr_1ct[i_min_diff]]), 
                        np.log10(ec_real[index_arr_1ct[i_min_diff][0]]),
                    c=colors[index_arr_1ct[i_min_diff]],#'tab:blue',
                    alpha=0.6, edgecolor='face')
        ax[0].axline((-2, -2), (3, 3), c='k', ls='--')
        ax[0].set_title(r'$N_{CT} = 1$')
        ax[0].text(0.05, 0.9, f'RMSE = {rmse_arr_1ct[i_min_diff][0]:.2f}', transform=ax[0].transAxes)

        colors = np.where(production_ct_arr_nct[i_min_diff]==1, 'b', 
                np.where(production_ct_arr_nct[i_min_diff]==2, 'g',
                        np.where(production_ct_arr_nct[i_min_diff]==3, 'darkorchid',
                                    np.where(production_ct_arr_nct[i_min_diff]==4, 'saddlebrown',
                                            np.where(production_ct_arr_nct[i_min_diff]==5, 'goldenrod',
                                                        np.where(production_ct_arr_nct[i_min_diff]==6, 'black', 'tab:pink'))))))
        labels = np.where(production_ct_arr_nct[i_min_diff]==1, 'CT1', 
                        np.where(production_ct_arr_nct[i_min_diff]==2, 'CT2',
                                np.where(production_ct_arr_nct[i_min_diff]==3, 'CT3',
                                            np.where(production_ct_arr_nct[i_min_diff]==4, 'CT4',
                                                    np.where(production_ct_arr_nct[i_min_diff]==5, 'CT5',
                                                             np.where(production_ct_arr_nct[i_min_diff]==6, 'CT6', 'CT7'))))))
        
        ax[1].scatter(np.log10(ec_pred_arr_nct[i_min_diff][index_arr_nct[i_min_diff]]), 
                        np.log10(ec_real[index_arr_nct[i_min_diff][0]]),
                    c=colors[index_arr_nct[i_min_diff]], marker='o',
                    alpha=0.6, edgecolor='face')

        ax[1].axline((-2, -2), (3, 3), c='k', ls='--')
        ax[1].set_title(r'Worst of $N_{CT} = $'+str(n_ct))
        ax[1].text(0.5, 0.9, f'RMSE = {rmse_arr_nct[i_min_diff][0]:.2f}', transform=ax[1].transAxes)

        #### Maximum difference
        colors = np.where(production_ct_arr_nct[i_max_diff]==1, 'b', 
                np.where(production_ct_arr_nct[i_max_diff]==2, 'g',
                        np.where(production_ct_arr_nct[i_max_diff]==3, 'darkorchid',
                                    np.where(production_ct_arr_nct[i_max_diff]==4, 'saddlebrown',
                                            np.where(production_ct_arr_nct[i_max_diff]==5, 'goldenrod',
                                                        np.where(production_ct_arr_nct[i_max_diff]==6, 'black', 'tab:pink'))))))
        labels = np.where(production_ct_arr_nct[i_max_diff]==1, 'CT1', 
                        np.where(production_ct_arr_nct[i_max_diff]==2, 'CT2',
                                np.where(production_ct_arr_nct[i_max_diff]==3, 'CT3',
                                            np.where(production_ct_arr_nct[i_max_diff]==4, 'CT4',
                                                    np.where(production_ct_arr_nct[i_max_diff]==5, 'CT5',
                                                             np.where(production_ct_arr_nct[i_max_diff]==6, 'CT6', 'CT7'))))))
        
        ax[2].scatter(np.log10(ec_pred_arr_nct[i_max_diff][index_arr_nct[i_max_diff]]), 
                        np.log10(ec_real[index_arr_nct[i_max_diff][0]]),
                    c=colors[index_arr_nct[i_max_diff]], marker='o',
                    alpha=0.6, edgecolor='face')

        ax[2].axline((-2, -2), (3, 3), c='k', ls='--')
        ax[2].text(0.6, 0.1, f'RMSE = {rmse_arr_nct[i_max_diff][0]:.2f}', transform=ax[2].transAxes)
        ax[2].set_title(r'Best of $N_{CT} = $'+str(n_ct))

        f.supxlabel(r'$log_{10}\ Predicted\ metabolome$', fontsize=15)
        f.supylabel(r'$log_{10}\ Empirical\ metabolome$', fontsize=15)
        f.tight_layout()

        if figsave_flag:
            f.figure.savefig(fig_path+'/metabolome-prediction-comparison-npro-'+str(npro)+'.png', dpi=300, bbox_inches='tight')
            plt.close(f.figure)
        else:
            plt.show()

#### Pooled pair plots
rmse_pooled_1ct, rmse_pooled_nct, balance_flag_pooled, rep_num = [[]], [[]], [[]], [[]]
slope_df, sublinear_cell_lines = pd.read_pickle(home_dir + '/cancer_power_law_stats.pickle')
for cl in sublinear_cell_lines:
    pickle_path = '../raw-output/2-celltypes/no-learn-balanced-net/'+cl

    [balance_arr_1ct, balance_arr_nct, balance_flag_arr, balanced_networks_list,
                        rmse_arr_1ct, rmse_arr_nct,
                        ec_pred_arr_1ct, ec_pred_arr_nct,
                        index_arr_1ct, index_arr_nct,
                        production_ct_arr_1ct, production_ct_arr_nct] = pd.read_pickle(pickle_path + '/balanced-networks.pickle')
    rmse_pooled_1ct.append(rmse_arr_1ct)
    rmse_pooled_nct.append(rmse_arr_nct)
    balance_flag_pooled.append(balance_flag_arr)
    rep_num.append(np.arange(1, len(rmse_arr_1ct)+1))

rmse_pooled_1ct = np.array(rmse_pooled_1ct[1:])
rmse_pooled_nct = np.array(rmse_pooled_nct[1:])
balance_flag_pooled = np.array(balance_flag_pooled[1:])
rep_num = np.array(rep_num[1:])

n_rand = len(rmse_arr_1ct)

df_pooled = pd.DataFrame({'Replicate': np.concatenate([np.arange(1, 3101), np.arange(1, 3101)]).ravel(),
                'N_CT': np.array([np.ones_like(rmse_pooled_1ct), np.zeros_like(rmse_pooled_1ct)+2]).ravel().astype(int),
                'Balance': np.concatenate([balance_flag_pooled, balance_flag_pooled]).ravel(),
                'RMSE': np.concatenate([rmse_pooled_1ct, rmse_pooled_nct]).ravel()})

g, ax = plt.subplots(1, 1, figsize = (3, 5))
ax = sns.lineplot(data=df_pooled, x='N_CT', y='RMSE',
                units='Replicate', hue='Balance',
                palette='coolwarm_r', hue_norm=(0, 1),
                dashes=False, estimator=None, zorder=1, legend=True)
ax = sns.scatterplot(data=df_pooled, x='N_CT', y='RMSE',
                    hue='Balance', palette='coolwarm_r', hue_norm=(0, 1),
                    edgecolor='face', alpha=0.8, zorder=2, legend=False)
ax.set_xlabel(r'$N_{CT}$')
ax.set(xlim=(0.5, 2.5), xticks=[1, 2])
ax.get_legend().set_title('Balance')

net_state = 'no-learn-balanced-net/'
figsave_flag = 1
fig_path = '../figures/2-celltypes/'+net_state
if figsave_flag:
    g.figure.savefig(fig_path+'/balance-rmse-comparison-all-cell-lines-pooled.png', dpi=300, bbox_inches='tight')
    plt.close(g.figure)
else:
    plt.show()

# %%
########## Figure 3
figsave_flag = 1
rmse_arr_heatmap = [[]]
num_final_heatmap = [[]]

for cl in sublinear_cell_lines:
    rmse = []
    num_final = []
    ec_real = ec_metabolome.loc[:, cl].values
    for n_ct in range(2, 8):
        # cl='A549-ATCC'
        pickle_path = '../raw-output/'+str(n_ct)+'-celltypes/no-learn-balanced-net/'+cl

        [balance_arr_1ct, balance_arr_nct, balance_flag_arr, balanced_networks_list,
                            rmse_arr_1ct, rmse_arr_nct,
                            ec_pred_arr_1ct, ec_pred_arr_nct,
                            index_arr_1ct, index_arr_nct,
                            production_ct_arr_1ct, production_ct_arr_nct] = pd.read_pickle(pickle_path + '/balanced-networks.pickle')
        rmse.append(rmse_arr_nct[balance_flag_arr].mean())
        num_final.append(balance_flag_arr.sum())

        f, ax = plt.subplots(1, 1, figsize=(4, 3.5))

        rmse_diff = rmse_arr_1ct - rmse_arr_nct
        i_min_diff = np.where(rmse_diff == rmse_diff.min(), True, False)
        i_max_diff = np.where(rmse_diff == rmse_diff.max(), True, False)

        colors = np.where(production_ct_arr_nct[i_max_diff]==1, 'b', 
                np.where(production_ct_arr_nct[i_max_diff]==2, 'g',
                        np.where(production_ct_arr_nct[i_max_diff]==3, 'darkorchid',
                                    np.where(production_ct_arr_nct[i_max_diff]==4, 'saddlebrown',
                                            np.where(production_ct_arr_nct[i_max_diff]==5, 'goldenrod',
                                                        np.where(production_ct_arr_nct[i_max_diff]==6, 'black', 'tab:pink'))))))
        labels = np.where(production_ct_arr_nct[i_max_diff]==1, 'CT1', 
                        np.where(production_ct_arr_nct[i_max_diff]==2, 'CT2',
                                np.where(production_ct_arr_nct[i_max_diff]==3, 'CT3',
                                            np.where(production_ct_arr_nct[i_max_diff]==4, 'CT4',
                                                    np.where(production_ct_arr_nct[i_max_diff]==5, 'CT5',
                                                             np.where(production_ct_arr_nct[i_max_diff]==6, 'CT6', 'CT7'))))))
        
        f = plt.scatter(np.log10(ec_pred_arr_nct[i_max_diff][index_arr_nct[i_max_diff]]), 
                                np.log10(ec_real[index_arr_nct[i_max_diff][0]]),
                            c=colors[index_arr_nct[i_max_diff]], marker='o',
                            alpha=0.6, edgecolor='face')

        plt.axline((-2, -2), (3, 3), c='k', ls='--')
        plt.text(0.6, 0.1, f'RMSE = {rmse_arr_nct[i_max_diff][0]:.2f}', transform=f.axes.transAxes)
        plt.title(cl+str('; ')+r'$N_{CT} = $'+str(n_ct))
        plt.xlabel(r'$log_{10}\ Predicted\ metabolome$', fontsize=15)
        plt.ylabel(r'$log_{10}\ Empirical\ metabolome$', fontsize=15)

        net_state = 'no-learn-balanced-net/'
        figsave_flag = 1
        fig_path = '../figures/'+str(n_ct)+'-celltypes/'+net_state+cl
        if figsave_flag:
            f.figure.savefig(fig_path+'/best-performing-network-prediction.png', dpi=300, bbox_inches='tight')
            plt.close(f.figure)
        else:
            plt.show()

    rmse_arr_heatmap.append(rmse)
    num_final_heatmap.append(num_final)

rmse_arr_heatmap = np.array(rmse_arr_heatmap[1:])
num_final_heatmap = np.array(num_final_heatmap[1:])

heatmap_df = pd.DataFrame(rmse_arr_heatmap,
                          columns=np.arange(2, 8),
                          index=sublinear_cell_lines)

f, ax = plt.subplots(1, 1, figsize=(7.5, 15))
ax = sns.heatmap(data=heatmap_df, cmap='crest')
ax.set_xlabel(r'$N_{CT}$')
ax.set_ylabel('Cell line')
ax.set_title(r'RMSE vs $N_{CT}$')

figsave_flag=1
if figsave_flag:
    f.figure.savefig('../figures/no-learn-rmse-vs-nct-heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(f.figure)
else:
    plt.show()

with sns.axes_style("darkgrid"):
    g = sns.lineplot(data=heatmap_df.T, palette='Blues_d', legend=False)
    g.set_xlabel(r'$N_{CT}$')
    g.set_ylabel('RMSE')
    g.set_title('Random networks w/o learning', pad=12)
if figsave_flag:
    g.figure.savefig('../figures/no-learn-rmse-vs-nct-lineplot.png', dpi=300, bbox_inches='tight')
    plt.close(g.figure)
else:
    plt.show()


# %%
##### Figure 4
for k in tqdm(range(len(sublinear_cell_lines)), desc='Cell line: '):#np.arange(0, 1):
        for n_ct in np.arange(3, 6):
            figsave_flag = 1
            reward = 0.5
            penalty = 0.5
            cl = sublinear_cell_lines[k]
            ec_real = ec_metabolome.loc[:, cl]

            pickle_path = '../raw-output/'+str(n_ct)+'-celltypes/optim-net/'+cl

            [x_ori_list, x_optim_list, error_plot_list, log_bias_list, log_bias_combined_list, n_pred_list, residual_list, balance_flag_list, prod_overlap_list, con_overlap_list,
                        metabolome_pred_before_list, metabolome_meas_before_list,
                        metabolome_pred_after_list, metabolome_meas_after_list,
                        valid_index_before_list, valid_index_after_list] = pd.read_pickle(pickle_path + '/reward-'+str(reward)+'-penalty-'+str(penalty)+'-optimised_network_output.pickle')
            max_links = int(len(x_ori_list[0].flatten())/2)
            n_reps = len(balance_flag_list)

            init_pred_error = np.array([arr[0] for arr in log_bias_list])
            final_pred_error = np.array([arr[-1] for arr in log_bias_list])

            init_residual = np.array([arr[0] for arr in residual_list])
            final_residual = np.array([arr[-1] for arr in residual_list])

            i_best_net = np.where(final_pred_error == final_pred_error.min())[0] #np.where(pred_error_change == pred_error_change.max())[0]
            x_ori_best = x_ori_list[i_best_net].flatten()
            x_optim_best = x_optim_list[i_best_net].flatten()

            fig = plt.figure(figsize=(9, 6))
            gs = GridSpec(2, 2, figure=fig)
            ax1 = fig.add_subplot(gs[:, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)

            #### A quick glance of where sims have begun and ended
            # colors = np.where(balance_flag_list, 'b', 'tab:red')
            for i in range(n_reps):
                # sns.lineplot(log_bias_list[i], ax=ax1, color=colors[i])
                log_bias = np.array(log_bias_list[i])
                norm_log_bias = log_bias/log_bias.sum()
                lines = colored_line(con_overlap_list[i], prod_overlap_list[i],
                                    c=norm_log_bias, ax=ax1, cmap='Greens', alpha=0.6, zorder=1)

            con_overlap_final = np.array([arr[-1] for arr in con_overlap_list])
            prod_overlap_final = np.array([arr[-1] for arr in prod_overlap_list])
            final_error = np.array([arr[-1] for arr in log_bias_list])
            ax1.scatter(x=con_overlap_final, y=prod_overlap_final,
                        s=35, c=final_error, cmap='Greens', alpha=0.9, zorder=2)
            ax1.set_xlabel("# consumption overlap")
            ax1.set_ylabel("# production overlap")
            ax1.set_title("%d replicate runs" % n_reps)
            ax1.set_xlim(0, 60)
            ax1.set_ylim(0, 22)
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

            ##### Check out the best performing network
            p_arr_ori = np.array([i*j for i, j in zip(x_ori_best[max_links:].reshape(n_ct, -1), np.arange(1, n_ct+1).tolist())]).sum(0)
            p_arr_optim = np.array([i*j for i, j in zip(x_optim_best[max_links:].reshape(n_ct, -1), np.arange(1, n_ct+1).tolist())]).sum(0)

            met_pred_before = metabolome_pred_before_list[i_best_net][0].astype(np.float64)
            met_pred_after = metabolome_pred_after_list[i_best_net][0].astype(np.float64)

            met_meas_before = metabolome_meas_before_list[i_best_net][0].astype(np.float64)
            met_meas_after = metabolome_meas_after_list[i_best_net][0].astype(np.float64)

            i_before = valid_index_before_list[i_best_net][0].astype(bool)
            i_after = valid_index_after_list[i_best_net][0].astype(bool)

            c_before = np.where(p_arr_ori == 0, 'tab:gray', np.where(p_arr_ori == 1, 'b', 
                                np.where(p_arr_ori == 2, 'g',
                                        np.where(p_arr_ori == 3, 'darkorchid', 
                                                np.where(p_arr_ori == 4, 'saddlebrown', 
                                                        np.where(p_arr_ori == 5, 'goldenrod',
                                                                    np.where(p_arr_ori == 6, 'black', 'tab:pink')))))))                     
            # c_before = np.where(i_before, c_before, 'tab:gray')

            c_after = np.where(p_arr_optim == 0, 'tab:gray', np.where(p_arr_optim == 1, 'b', 
                                np.where(p_arr_optim == 2, 'g',
                                        np.where(p_arr_optim == 3, 'darkorchid', 
                                                np.where(p_arr_optim == 4, 'saddlebrown', 
                                                        np.where(p_arr_optim == 5, 'goldenrod',
                                                                    np.where(p_arr_optim == 6, 'black', 'tab:pink')))))))
            # c_after = np.where(i_after, c_after, 'tab:gray')

            ori_list = x_ori_best[max_links:].reshape(n_ct, -1)
            overlap_colours_before = np.array([np.where(ori_list[:, i].sum() > 1, 'overlap', 'no') for i in range(115)])
            c_before[overlap_colours_before == 'overlap'] = 'tab:red'


            optim_list = x_optim_best[max_links:].reshape(n_ct, -1)
            overlap_colours_after = np.array([np.where(optim_list[:, i].sum() > 1, 'overlap', 'no') for i in range(115)])
            c_after[overlap_colours_after == 'overlap'] = 'tab:red'

            a_before = np.where(i_before, 0.8, 0.1)
            a_after = np.where(i_after, 0.8, 0.)
            a_after[overlap_colours_after == 'no'] = 0.2
            a_after[overlap_colours_after == 'overlap'] = 1

            s_after = np.where(overlap_colours_after == 'overlap', 80, 40)

            edgecolor_before = np.where(overlap_colours_before == 'overlap', 'k', c_before)
            edgecolor_after = np.where(overlap_colours_after == 'overlap', 'k', c_after)

            # fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 7))
            ax2.scatter(np.log10(met_pred_before), np.log10(met_meas_before), c=c_before, alpha=a_before, s=50, edgecolors=edgecolor_before)
            ax2.axline((-2, -2), (3, 3), c='k', transform=ax2.transAxes)
            ax2.text(1.02, 0.5, 'Original', size=SMALL_SIZE, rotation='vertical', va='center', transform=ax2.transAxes)
            ax2.text(0.05, 0.8, r'$\chi_{excess}=$'+f'{init_residual[i_best_net][0].round(decimals=3):.2f}', transform=ax2.transAxes)
            # ax2.set_xlabel(r'$log_{10}\ Predicted\ metabolome$')
            ax2.set_ylabel(r'$log_{10}\ Data$', size=SMALL_SIZE)
            ax2.text(0.05, 0.9, f'RMSE = {init_pred_error[i_best_net][0].round(decimals=3):.2f}', transform=ax2.transAxes)

            ax3.scatter(np.log10(met_pred_after), np.log10(met_meas_after), c=c_after, alpha=a_after, s=s_after, edgecolors=edgecolor_after)
            ax3.axline((-2, -2), (2, 2), c='k', transform=ax3.transAxes)
            ax3.text(1.02, 0.5, 'Optimised', size=SMALL_SIZE, rotation='vertical', va='center', transform=ax3.transAxes)
            ax3.text(0.05, 0.9, f'RMSE = {final_pred_error[i_best_net][0].round(decimals=3):.2f}', transform=ax3.transAxes)
            ax3.text(0.05, 0.8, r'$\chi_{excess}=$'+f'{final_residual[i_best_net][0].round(decimals=3):.2f}', transform=ax3.transAxes)
            ax3.set_xlabel(r'$log_{10}\ Prediction$')
            ax3.set_ylabel(r'$log_{10}\ Data$', size=SMALL_SIZE)

            fig.suptitle(f'Network optimisation for {cl} with reward {reward}')
            fig.tight_layout(pad=0.6)


            fig_path = '../figures/'+str(n_ct)+'-celltypes/optim-net/'+cl
            if figsave_flag:
                try:
                    os.makedirs(fig_path)
                except:
                    pass
                fig.savefig(fig_path+'/reward-'+str(reward)+'-penalty-'+str(penalty)+'-network-predictions.png', dpi=300)
                plt.close(fig)

            #### More overlap means less error?
            num_prod_overlap = []
            num_con_overlap = []
            for x in x_optim_list:
                ## Production overlap
                optim_list = x[max_links:].reshape(n_ct, -1)
                prod_overlap = np.array([optim_list[:, i].sum() for i in range(115)])
                num_prod_overlap.append(len(prod_overlap[prod_overlap > 1]))
                
                ## Consumption overlap
                optim_list = x[:max_links].reshape(n_ct, -1)
                con_overlap = np.array([optim_list[:, i].sum() for i in range(115)])
                num_con_overlap.append(len(con_overlap[con_overlap > 1]))
            num_con_overlap, num_prod_overlap = np.array(num_con_overlap), np.array(num_prod_overlap)
            
            overlap_df = pd.DataFrame({'LinkType': np.concatenate([np.repeat('Consumption', len(num_con_overlap)), np.repeat('Production', len(num_prod_overlap))]),
                                       'Balance': np.concatenate([balance_flag_list, balance_flag_list]),
                                       'Overlap': np.concatenate([num_con_overlap, num_prod_overlap]),
                                       'RMSE': np.concatenate([final_error, final_error])})
            with sns.axes_style('darkgrid'):
                f = sns.lmplot(data=overlap_df, x='Overlap', y='RMSE', row='LinkType',
                           hue='Balance', palette={False: 'r', True: 'b'},
                           line_kws={'linewidth': 2}, scatter_kws={'s': 15},
                           facet_kws=dict(sharex=False))
                f.set_xlabels('')
                f.figure.supxlabel('# overlapping metabolites', size=MEDIUM_SIZE)
                # f.figure.tight_layout()

            if figsave_flag:
                f.savefig(fig_path+'/reward-'+str(reward)+'-penalty-'+str(penalty)+'-overlap-vs-rmse-lmplot.png', dpi=300)
                plt.close(f.figure)
            else:
                plt.show()            


            ###### Which overlaps are learnt, which metabolites have overlap either in production or consumption across replicates for each cell line
            consumption_overlap_indices, consumption_overlap_mets, production_overlap_indices, production_overlap_mets = [[]], [[]], [[]], [[]]
            
            max_links = len(x_optim_best)//2
            met_ID = diet.index.to_numpy()
            for x in x_optim_list:
                clist, cmets, plist, pmets = calculate_overlap_stats(x, n_ct, max_links, met_ID)
                consumption_overlap_indices.append(clist)
                consumption_overlap_mets.append(cmets)
                production_overlap_indices.append(plist)
                production_overlap_mets.append(pmets)

            consumption_overlap_indices = np.concatenate(consumption_overlap_indices)
            consumption_overlap_mets = np.concatenate(consumption_overlap_mets).astype(int)
            production_overlap_indices = np.concatenate(production_overlap_indices)
            production_overlap_mets = np.concatenate(production_overlap_mets).astype(int)

            ### Overlapping links
            overlap_indices_df = pd.DataFrame({'LinkType': np.concatenate([['Consumption']*len(consumption_overlap_indices), ['Production']*len(production_overlap_indices)]),
                                            'Overlap': np.concatenate([consumption_overlap_indices, production_overlap_indices])})
            
            unique_indices = np.unique(overlap_indices_df['Overlap'].values) # All unique overlaps
            index_len = np.array([len(i) for i in unique_indices]) # How many celltypes in each overlap
            unique_indices_sorted = [[]]
            for i in range(3, n_ct + n_ct - 1 + 1): # Range limits based on the string length for n_ct celltype overlap e.g., '1+2' is a string of length 3, '1+2+3' is of length 5, so this goes as n + (n-1), where n is the number of overlapping celltypes. The extra plus one is to account for the range function not including the last number
                if len(unique_indices[index_len == i]) > 0: # If there is an overlap of this length
                    unique_indices_sorted.append(unique_indices[index_len == i]) # Retrieving all overlaps of a given length this way sorts the unique elements of that particular length
            unique_indices_sorted = np.concatenate(unique_indices_sorted[1:])

            overlap_indices_df['Overlap'] = pd.Categorical(overlap_indices_df['Overlap'], categories=unique_indices_sorted) # This sets the column to categorical with the levels in the order above, giving 2-celltype overlaps first, then three and so on, and sorted by celltype number within each overlap length
            with sns.axes_style('darkgrid'):
                g = sns.displot(data=overlap_indices_df, x='Overlap',
                                hue='LinkType', palette={'Consumption': 'tab:green',
                                                        'Production': 'tab:blue'},
                                multiple='stack', stat='probability', common_norm=False, 
                                discrete=True, shrink=0.9, height=4, aspect=3)
                g.set_xticklabels(rotation=65)
                g.set_xlabels('Overlapping celltypes')
                sns.move_legend(g, "lower center", bbox_to_anchor=(0.65, 0.75), ncol=2,
                                title='Link type', frameon=True)
                
            if figsave_flag:
                g.savefig(fig_path+'/reward-'+str(reward)+'-penalty-'+str(penalty)+'-overlapping-links.png', dpi=300)
                plt.close(g.figure)
            else:
                plt.show()

            ### Overlapping mets
            overlap_mets_df = pd.DataFrame({'LinkType': np.concatenate([['Consumption']*len(consumption_overlap_mets), ['Production']*len(production_overlap_mets)]),
                                            'Metabolite': np.concatenate([consumption_overlap_mets, production_overlap_mets])})
            mets_order = np.array(ec_real.sort_values(inplace=False).index) # IDs of overlapping metabolites sorted by their abundances in the fresh medium
            mets_ranks = np.arange(len(mets_order)) # Assigning ranks to the sorted metabolites
            overlap_mets_df.loc[:, 'MetRanks'] = np.array([mets_ranks[np.where(i == mets_order)[0]][0] for i in overlap_mets_df['Metabolite'].values]) # Mapping metabolite IDs in the dataframe to ranks based on fresh medium abundaces
            with sns.axes_style('darkgrid'): # Histogram of ranks
                h = sns.displot(data=overlap_mets_df, x='MetRanks',
                                hue='LinkType', palette={'Consumption': 'tab:green',
                                                        'Production': 'tab:blue'},
                                multiple='stack', stat='probability', common_norm=False,
                                discrete=True, height=3.85, aspect=3.5, legend=True)
                h.set_xlabels('Abundance rank')
                sns.move_legend(h, "lower center", bbox_to_anchor=(0.67, 0.7), ncol=2,
                                title='Link type', frameon=True)
                plt.xlim(-5, 120)
            if figsave_flag:
                h.savefig(fig_path+'/reward-'+str(reward)+'-penalty-'+str(penalty)+'-overlapping-metabolites.png', dpi=300)
                plt.close(h.figure)
            else:
                plt.show()

            ### Which overlapping links are common among the metabolites with the highest learnt overlap
            ### Consumption
            top_met_consumption = overlap_mets_df[overlap_mets_df['LinkType'] == 'Consumption']['Metabolite'].value_counts().reset_index().iloc[0, 0]
            top_met_consumption_links = overlap_indices_df[(overlap_indices_df['LinkType']=='Consumption')*(overlap_mets_df['Metabolite']==top_met_consumption)]

            ### Production
            top_met_production = overlap_mets_df[overlap_mets_df['LinkType'] == 'Production']['Metabolite'].value_counts().reset_index().iloc[0, 0]
            top_met_production_links = overlap_indices_df[(overlap_indices_df['LinkType']=='Production')*(overlap_mets_df['Metabolite']==top_met_production)]

            top_met_links_df = pd.concat([top_met_consumption_links, top_met_production_links])
            top_met_unique_indices = np.unique(top_met_links_df['Overlap'].values) # All unique overlaps
            index_len = np.array([len(i) for i in top_met_unique_indices]) # How many celltypes in each overlap
            unique_indices_sorted = [[]]
            for i in range(3, n_ct + n_ct - 1): # Range limits based on the string length for n_ct celltype overlap e.g., '1+2' is a string of length 3, '1+2+3' is of length 5, so this goes as n + (n-1), where n is the number of overlapping celltypes
                if len(top_met_unique_indices[index_len == i]) > 0: # If there is an overlap of this length
                    unique_indices_sorted.append(top_met_unique_indices[index_len == i]) # Retrieving all overlaps of a given length this way sorts the unique elements of that particular length
            unique_indices_sorted = np.concatenate(unique_indices_sorted[1:])
            top_met_links_df['Overlap'] = pd.Categorical(top_met_links_df['Overlap'], categories=unique_indices_sorted)
            with sns.axes_style('darkgrid'):
                q = sns.displot(data=top_met_links_df, x='Overlap',
                                hue='LinkType', palette={'Consumption': 'tab:green',
                                                        'Production': 'tab:blue'},
                                multiple='stack', stat='probability', common_norm=False, 
                                discrete=True, shrink=0.9, height=4, aspect=3)
                q.set_xticklabels(rotation=65)
                q.set_xlabels('Overlapping celltypes')
                q.figure.suptitle('Top metabolites overlap', y=1.03)
                sns.move_legend(q, "lower center", bbox_to_anchor=(0.67, 0.7), ncol=2,
                                title='Link type', frameon=True)
            if figsave_flag:
                q.savefig(fig_path+'/reward-'+str(reward)+'-penalty-'+str(penalty)+'-top-mets-overlapping-links.png', dpi=300)
                plt.close(q.figure)
            else:
                plt.show()


# %%
##### Heatmaps with learning
figsave_flag = 0
rmse_arr_heatmap = [[]]
num_final_heatmap = [[]]
for cl in sublinear_cell_lines:
    rmse = []
    num_final = []
    reward = 0.1
    penalty = 0.1
    for n_ct in range(3, 6):
        # cl='A549-ATCC'
        pickle_path = '../raw-output/'+str(n_ct)+'-celltypes/optim-net/'+cl

        [x_ori_list, x_optim_list, error_plot_list, log_bias_list, log_bias_combined_list, n_pred_list, residual_list, balance_flag_list, prod_overlap_list, con_overlap_list,
                        metabolome_pred_before_list, metabolome_meas_before_list,
                        metabolome_pred_after_list, metabolome_meas_after_list,
                        valid_index_before_list, valid_index_after_list] = pd.read_pickle(pickle_path + '/reward-'+str(reward)+'-penalty-'+str(penalty)+'-optimised_network_output.pickle')
        final_error = np.array([arr[-1] for arr in log_bias_list])
        rmse.append(final_error[balance_flag_list].mean())
        num_final.append(balance_flag_list.sum())

    rmse_arr_heatmap.append(rmse)
    num_final_heatmap.append(num_final)

rmse_arr_heatmap = np.array(rmse_arr_heatmap[1:])
num_final_heatmap = np.array(num_final_heatmap[1:])

learn_heatmap_df = pd.DataFrame(rmse_arr_heatmap,
                          columns=np.arange(3, 6),
                          index=sublinear_cell_lines)

f, ax = plt.subplots(1, 1, figsize=(7.5, 15))
ax = sns.heatmap(data=learn_heatmap_df, cmap='crest')
ax.set_xlabel(r'$N_{CT}$')
ax.set_ylabel('Cell line')
ax.set_title(r'RMSE vs $N_{CT}$')

if figsave_flag:
    f.figure.savefig('../figures/optim-rmse-vs-nct-heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(f.figure)
else:
    plt.show()

with sns.axes_style("darkgrid"):
    g = sns.lineplot(data=learn_heatmap_df.T, palette='Blues_d', legend=False)
    g.set_xlabel(r'$N_{CT}$')
    g.set_ylabel('RMSE')
    g.set_title('Random networks w/o learning', pad=12)
if figsave_flag:
    g.figure.savefig('../figures/optim-rmse-vs-nct-lineplot.png', dpi=300, bbox_inches='tight')
    plt.close(g.figure)
else:
    plt.show()

# %%
######### Figure 2, as in slide 7 onwards
num_pred_init = np.array([arr[1] for arr in n_pred_list])
init_error = np.array([arr[0] for arr in log_bias_list])

num_pred_final = np.array([arr[-1] for arr in n_pred_list])
final_error = np.array([arr[-1] for arr in log_bias_list])
init_centroid = np.array([num_pred_init.mean(), init_error.mean()])
final_centroid = np.array([num_pred_final.mean(), final_error.mean()])
v = final_centroid - init_centroid

sparse_init = np.array([arr.sum()/len(arr) for arr in x_ori_list])
sparse_final = np.array([arr.sum()/len(arr) for arr in x_optim_list])

points_df = pd.DataFrame({'NetState': np.repeat(['Initial', 'Final'], n_reps),
              'NumPred': np.concatenate([num_pred_init, num_pred_final]),
               'RMSE': np.concatenate([init_error, final_error]),
               'Sparsity': np.concatenate([sparse_init, sparse_final])})

ax = sns.jointplot(data=points_df, x='NumPred', y='RMSE',
                    hue='NetState', palette='crest',
                    s=30, alpha=1, zorder=2)
ax = plt.plot(init_centroid[0], init_centroid[1], zorder=3,
              color=sns.cm.crest(0.1), markeredgecolor='k', marker='X', markersize=10)
ax = plt.plot(final_centroid[0], final_centroid[1], zorder=3,
              color=sns.cm.crest(0.8), markeredgecolor='k', marker='X', markersize=10)

colors = plt.cm.gist_yarg(np.linspace(0, 0.7, 100))
for i in range(n_reps):
    ax = sns.lineplot(x=n_pred_list[i][1:], y=log_bias_list[i],
                estimator=None, c=colors[i], linewidth=1.5,
                sort=False, alpha=0.5, zorder=1)
ax.quiver(init_centroid[0], init_centroid[1], v[0], v[1],
          angles='xy', scale_units='xy', scale=1.1, zorder=2,
          width=0.015,
          headwidth=3, headlength=5, headaxislength=5, color='tab:red', alpha=0.9,
          edgecolor='k', linewidth=0.7)
# plt.text(x=0.55, y=0.3, s=r'X$\hat{\imath}$ + Y$\hat{\jmath}$',
#          fontdict={'color': 'tab:red', 'rotation': 0}, transform=ax.transAxes) 
plt.show()

g = sns.JointGrid(x=num_pred_init, y=init_error,
                  xlim=(num_pred_init.min()-1, num_pred_init.max()+1),
                  ylim=(init_error.min()-0.25, init_error.max()+0.25),
                  marginal_ticks=False, space=0.01)
g.plot_joint(sns.regplot, color='k',
            line_kws={'color': 'tab:red', 'linewidth': 2}, scatter_kws={'s': 25, 'edgecolor': 'k', 'alpha': 0.6})
g.plot_marginals(sns.kdeplot, fill=True, color='tab:red')
rho, pval = spearmanr(num_pred_init, init_error)

g.ax_joint.text(x=0.25, y=0.03, s=f'Spearman\'s $\\rho$ = {rho:.2f}', transform=g.ax_joint.transAxes)
g.ax_joint.set_xlabel('# metabolites predicted')
g.ax_joint.set_ylabel('RMSE')
g.figure.suptitle('Initial network prediction')
g.figure.tight_layout()

g = sns.JointGrid(x=num_pred_final, y=final_error,
                  xlim=(num_pred_final.min()-1, num_pred_final.max()+1),
                  ylim=(final_error.min()-0.25, final_error.max()+0.25),
                  marginal_ticks=False, space=0.01)
g.plot_joint(sns.regplot, color='k',
            line_kws={'color': 'tab:red', 'linewidth': 2}, scatter_kws={'s': 25, 'edgecolor': 'k', 'alpha': 0.6})
g.plot_marginals(sns.kdeplot, fill=True, color='tab:red')
rho, pval = spearmanr(num_pred_final, final_error)

g.ax_joint.text(x=0.25, y=0.03, s=f'Spearman\'s $\\rho$ = {rho:.2f}', transform=g.ax_joint.transAxes)
g.ax_joint.set_xlabel('# metabolites predicted')
g.ax_joint.set_ylabel('RMSE')
g.figure.suptitle('Final network prediction')
g.figure.tight_layout()


# %%
######### Sparsity as the proportion of present links
sns.kdeplot(data=points_df, x='Sparsity', hue='NetState', palette='crest', fill=True)
plt.xlabel('Proportion of present links')
plt.suptitle('Sparsity of networks; '+str(n_reps)+' replicates')
plt.title(f'Reward = {reward:.2f}; Penalty = {penalty:.2f}')
plt.tight_layout()


# %%
#### Vector components and other extra plots across all cell lines and reward-penalty param values; barplots at the end have been included in Figure 2 slides in network-figure-plan.pptx; remaining boxplots are dumped separately and are not yet part of the current figure plan.
home_dir = os.getcwd()
slope_df, sublinear_cell_lines = pd.read_pickle(home_dir + '/cancer_power_law_stats.pickle')

celltype_ID, celltypefreq, ec_metabolome_ID, ec_metabolome, met_baseline, core_mean = pd.read_pickle(home_dir + '/1-cells-data.pickle')
diet = met_baseline.mean(axis=1)


vector_all = [[]]
net_consumption_init = [[]]
net_consumption_final = [[]]
rmse_init = [[]]
rmse_final = [[]]
param_values = np.array([0.1, 0.5])

for p in param_values:

    reward = p
    penalty = p

    v_arr = []
    init_con_arr = []
    final_con_arr = []
    init_error_arr = []
    final_error_arr = []
    for cl in sublinear_cell_lines:
        pickle_path = '../raw-output/2-celltypes/optim-net/balance-partition/'+cl

        [x_ori_list, x_optim_list, error_plot_list, log_bias_list, n_pred_list,
                    metabolome_pred_before_list, metabolome_meas_before_list,
                    metabolome_pred_after_list, metabolome_meas_after_list,
                    valid_index_before_list, valid_index_after_list] = pd.read_pickle(pickle_path + '/reward-'+str(reward)+'-penalty-'+str(penalty)+'-optimised_network_output.pickle')
        n_reps = len(n_pred_list)

        num_pred_init = np.array([arr[1] for arr in n_pred_list])
        init_error = np.array([arr[0] for arr in log_bias_list])

        num_pred_final = np.array([arr[-1] for arr in n_pred_list])
        final_error = np.array([arr[-1] for arr in log_bias_list])

        num_pred_diff = num_pred_final - num_pred_init
        error_diff = final_error - init_error
        v = np.array([[i, j] for i, j in zip(num_pred_diff, error_diff)])
        v_arr.append(v)

        max_links = x_ori_list.shape[1]//2
        n_mets = len(diet.values)
        consumption_init = np.array([np.dot(x_ori[:max_links].reshape(-1, n_mets).sum(0), diet.values) for x_ori in x_ori_list])
        consumption_final = np.array([np.dot(x_optim[:max_links].reshape(-1, n_mets).sum(0), diet.values) for x_optim in x_optim_list])

        init_con_arr.append(consumption_init)
        final_con_arr.append(consumption_final)

        init_error_arr.append(init_error)
        final_error_arr.append(final_error)

    vector_all.append(v_arr)
    net_consumption_init.append(init_con_arr)
    net_consumption_final.append(final_con_arr)
    rmse_init.append(init_error_arr)
    rmse_final.append(final_error_arr)

vector_all = np.array(vector_all[1:])
net_consumption_init = np.array(net_consumption_init[1:])
net_consumption_final = np.array(net_consumption_final[1:])
rmse_init = np.array(rmse_init[1:])
rmse_final = np.array(rmse_final[1:])

consumption_df = pd.DataFrame({'Param': np.repeat(param_values, len(sublinear_cell_lines)*n_reps),
                               'Cell line': np.array([np.repeat(sublinear_cell_lines, n_reps)]*2).ravel(),
                               'Initial': net_consumption_init.ravel(),
                               'Final': net_consumption_final.ravel()})

rmse_df = pd.DataFrame({'Param': np.repeat(param_values, len(sublinear_cell_lines)*n_reps),
                               'Cell line': np.array([np.repeat(sublinear_cell_lines, n_reps)]*2).ravel(),
                               'Initial': rmse_init.ravel(),
                               'Final': rmse_final.ravel()})

diff_df = pd.DataFrame({'Param': np.repeat(param_values, len(sublinear_cell_lines)*n_reps),
                               'Cell line': np.array([np.repeat(sublinear_cell_lines, n_reps)]*2).ravel(),
                               'NumPred': vector_all[:, :, :, 0].ravel(),
                               'RMSE': vector_all[:, :, :, 1].ravel()})

con_df_long = consumption_df.melt(id_vars=['Param', 'Cell line'], value_vars=['Initial', 'Final'], value_name='Consumption', var_name='NetState')

rmse_df_long = rmse_df.melt(id_vars=['Param', 'Cell line'], value_vars=['Initial', 'Final'], value_name='RMSE', var_name='NetState')


for p in param_values:
    f, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax = sns.boxplot(data=con_df_long[con_df_long['Param']==p], x='Cell line', y='Consumption',
                     hue='NetState', palette='viridis')
    ax.tick_params(axis='x', labelrotation=70, labelsize=12)
    ax.set_title('Reward = '+str(p)+'; Penalty = '+str(p))
    f.tight_layout()
    # f.savefig('../figures/2-celltypes/optim-net/reward-'+str(p)+'-penalty-'+str(p)+'-change-in-net-consumption.png', dpi=300)
    plt.close(f)

combined_df_long = con_df_long.copy()
combined_df_long.loc[:, 'RMSE'] = rmse_df_long['RMSE'].values

for p in param_values:
    ax = sns.jointplot(data=combined_df_long[combined_df_long['Param']==p], x='Consumption', y='RMSE',
                       hue='NetState', palette='viridis',
                       s=30, alpha=1)
    ax.ax_joint.set_title('Reward = '+str(p)+'; Penalty = '+str(p))
    ax.figure.tight_layout()

f, ax = plt.subplots(1, 1, figsize=(15, 4))
ax = sns.boxplot(data=diff_df, x='Cell line', y='RMSE',
                 hue='Param', palette='viridis', fliersize=0)
ax.tick_params(axis='x', labelrotation=70, labelsize=12.5)
ax.legend(title='Reward/Penalty',ncols=4)
# f.savefig('../figures/2-celltypes/optim-net/rmse-vs-reward-penalty.png', dpi=300)

f, ax = plt.subplots(1, 1, figsize=(15, 4))
ax = sns.boxplot(data=diff_df, x='Cell line', y='NumPred',
                 hue='Param', palette='viridis', fliersize=0)
ax.tick_params(axis='x', labelrotation=70, labelsize=12.5)
ax.set_ylabel('# metabolites predicted')
ax.legend(title='Reward/Penalty',ncols=4)
# f.savefig('../figures/2-celltypes/optim-net/numpred-vs-reward-penalty.png', dpi=300)

for p in param_values:
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(17, 6))
    sns.barplot(data=diff_df[diff_df['Param']==p], x='Cell line', y='NumPred',
                estimator='mean', errorbar='sd', color='tab:red', alpha=0.9,
                ax=ax[0], capsize=0.1)
    ax[0].spines.top.set_visible(False)
    ax[0].spines.right.set_visible(False)
    ax[0].set_ylabel('X')

    sns.barplot(data=diff_df[diff_df['Param']==p], x='Cell line', y='RMSE',
                estimator='mean', errorbar='sd', color='tab:red', alpha=0.9,
                ax=ax[1], capsize=0.1)
    ax[1].spines.top.set_visible(False)
    ax[1].spines.right.set_visible(False)
    ax[1].tick_params(axis='x', labelrotation=70, labelsize=17)
    ax[1].set_ylabel('Y')

    fig.suptitle('Vector components; reward, penalty = '+str(p), fontsize=20)
    fig.tight_layout()
    # fig.savefig('../figures/2-celltypes/optim-net/reward-'+str(p)+'-penalty-'+str(p)+'-vector-components.png', dpi=300)




# %%
pickle_path = '../raw-output/2-celltypes/optim-net/A549-ATCC'
reward = 0.01
penalty = 0.01

[x_ori_list, x_optim_list, error_plot_list, log_bias_list, n_pred_list,
             metabolome_pred_before_list, metabolome_meas_before_list,
             metabolome_pred_after_list, metabolome_meas_after_list,
             valid_index_before_list, valid_index_after_list] = pd.read_pickle(pickle_path + '/reward-'+str(reward)+'-penalty-'+str(penalty)+'-optimised_network_output.pickle')
n_reps = len(n_pred_list)

num_pred_init = np.array([arr[1] for arr in n_pred_list])
init_error = np.array([arr[0] for arr in log_bias_list])

num_pred_final = np.array([arr[-1] for arr in n_pred_list])
final_error = np.array([arr[-1] for arr in log_bias_list])
init_centroid = np.array([num_pred_init.mean(), init_error.mean()])
final_centroid = np.array([num_pred_final.mean(), final_error.mean()])
v = final_centroid - init_centroid

sparse_init = np.array([arr.sum()/len(arr) for arr in x_ori_list])
sparse_final = np.array([arr.sum()/len(arr) for arr in x_optim_list])

points_df = pd.DataFrame({'NetState': np.repeat(['Initial', 'Final'], n_reps),
              'NumPred': np.concatenate([num_pred_init, num_pred_final]),
               'RMSE': np.concatenate([init_error, final_error]),
               'Sparsity': np.concatenate([sparse_init, sparse_final])})

ax = sns.jointplot(data=points_df, x='NumPred', y='RMSE',
                    hue='NetState', palette='crest',
                    s=30, alpha=1, zorder=2)
ax = plt.plot(init_centroid[0], init_centroid[1], zorder=3,
              color=sns.cm.crest(0.1), markeredgecolor='k', marker='X', markersize=10)
ax = plt.plot(final_centroid[0], final_centroid[1], zorder=3,
              color=sns.cm.crest(0.8), markeredgecolor='k', marker='X', markersize=10)

colors = plt.cm.gist_yarg(np.linspace(0, 0.7, 100))
for i in range(n_reps):
    ax = sns.lineplot(x=n_pred_list[i][1:], y=log_bias_list[i],
                estimator=None, c=colors[i], linewidth=1.5,
                sort=False, alpha=0.5, zorder=1)
ax.quiver(init_centroid[0], init_centroid[1], v[0], v[1],
          angles='xy', scale_units='xy', scale=1.1, zorder=2,
          width=0.015,
          headwidth=3, headlength=5, headaxislength=5, color='tab:red', alpha=0.9,
          edgecolor='k', linewidth=0.7)
# plt.text(x=0.55, y=0.3, s=r'X$\hat{\imath}$ + Y$\hat{\jmath}$',
#          fontdict={'color': 'tab:red', 'rotation': 0}, transform=ax.transAxes) 
plt.show()

##### Check out the best performing network
pred_error_change = np.array([list[0]-list[-1] for list in log_bias_list])
final_pred_error = np.array([arr[-1] for arr in log_bias_list])

i_best_net = np.where(final_pred_error == final_pred_error.min())[0]#np.where(pred_error_change == pred_error_change.max())[0]
x_ori_best = x_ori_list[i_best_net].flatten()
x_optim_best = x_optim_list[i_best_net].flatten()

met_pred_before = metabolome_pred_before_list[i_best_net][0].astype(np.float64)
met_pred_after = metabolome_pred_after_list[i_best_net][0].astype(np.float64)

met_meas_before = metabolome_meas_before_list[i_best_net][0].astype(np.float64)
met_meas_after = metabolome_meas_after_list[i_best_net][0].astype(np.float64)

i_before = valid_index_before_list[i_best_net][0].astype(bool)
i_after = valid_index_after_list[i_best_net][0].astype(bool)

c_before = np.where(i_before, 'b', 'k')
c_after = np.where(i_after, 'b', 'k')
a_before = np.where(i_before, 1, 0.25)
a_after = np.where(i_after, 1, 0.25)

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(7, 4))
ax[0].scatter(np.log10(met_pred_before), np.log10(met_meas_before), c=c_before, alpha=a_before, s=30)
# ax[0].scatter(np.log10(metabolome_pred_old+1e-7), np.log10(metabolome_measured_old+1e-7), c='k', s=9)
ax[0].axline((-2, -2), (3, 3), c='k')
ax[0].set_title('Original network')
# ax[0].set_xlabel(r'$log_{10}\ Predicted\ metabolome$')
ax[0].set_ylabel(r'$log_{10}\ Empirical\ data$')

ax[1].scatter(np.log10(met_pred_after), np.log10(met_meas_after), c=c_after, alpha=a_after, s=30)
# ax[1].scatter(np.log10(metabolome_pred+1e-7), np.log10(metabolome_measured+1e-7), c='k', s=9)
ax[1].axline((-2, -2), (2, 2), c='k')
ax[1].set_title('Optimised network')
# ax[1].set_xlabel(r'$log_{10}\ Predicted\ metabolome$')
# ax[1].set_ylabel(r'$log_{10}\ Empirical\ data$')
# ax[1].set_xlabel('', labelpad=0.1)
fig.supxlabel(r'$log_{10}\ Predicted\ metabolome$')
fig.tight_layout(pad=0.2)


# %%

home_dir = os.getcwd()
slope_df, sublinear_cell_lines = pd.read_pickle(home_dir + '/cancer_power_law_stats.pickle')

celltype_ID, celltypefreq, ec_metabolome_ID, ec_metabolome, met_baseline, core_mean = pd.read_pickle(home_dir + '/1-cells-data.pickle')
diet = met_baseline.mean(axis=1)

ct_arr = np.array([1, 2, 3, 4])
vector_all = [[]]
rmse_init = [[]]
rmse_final = [[]]
numpred_init = [[]]
numpred_final = [[]]

for cl in sublinear_cell_lines:
    v_arr = []
    init_error_arr = []
    final_error_arr = []
    init_np_arr = []
    final_np_arr = []

    for n_ct in ct_arr:
        reward = 0.5
        penalty = 0.5

        pickle_path = '../raw-output/'+str(n_ct)+'-celltypes/optim-net/'+cl

        [x_ori_list, x_optim_list, error_plot_list, log_bias_list, n_pred_list,
                    metabolome_pred_before_list, metabolome_meas_before_list,
                    metabolome_pred_after_list, metabolome_meas_after_list,
                    valid_index_before_list, valid_index_after_list] = pd.read_pickle(pickle_path + '/reward-'+str(reward)+'-penalty-'+str(penalty)+'-optimised_network_output.pickle')
        n_reps = len(n_pred_list)

        num_pred_init = np.array([arr[1] for arr in n_pred_list])
        init_error = np.array([arr[0] for arr in log_bias_list])

        num_pred_final = np.array([arr[-1] for arr in n_pred_list])
        final_error = np.array([arr[-1] for arr in log_bias_list])

        num_pred_diff = num_pred_final - num_pred_init
        error_diff = final_error - init_error
        v = np.array([[i, j] for i, j in zip(num_pred_diff, error_diff)])
        v_arr.append(v)

        init_error_arr.append(init_error)
        final_error_arr.append(final_error)
        init_np_arr.append(num_pred_init)
        final_np_arr.append(num_pred_final)

    
    vector_all.append(v_arr)
    rmse_init.append(init_error_arr)
    rmse_final.append(final_error_arr)
    numpred_init.append(init_np_arr)
    numpred_final.append(final_np_arr)

vector_all = np.array(vector_all[1:])
rmse_init = np.array(rmse_init[1:])
rmse_final = np.array(rmse_final[1:])
numpred_init = np.array(numpred_init[1:])
numpred_final = np.array(numpred_final[1:])

diff_df = pd.DataFrame({'Cell line': np.repeat(sublinear_cell_lines, len(ct_arr)*n_reps),
                               'N_CT': np.array([np.repeat(ct_arr, n_reps)]*len(sublinear_cell_lines)).ravel(),
                               'Del_NumPred': vector_all[:, :, :, 0].ravel(),
                               'Del_RMSE': vector_all[:, :, :, 1].ravel()})

rmse_df = pd.DataFrame({'Cell line': np.repeat(sublinear_cell_lines, len(ct_arr)*n_reps),
                               'N_CT': np.array([np.repeat(ct_arr, n_reps)]*len(sublinear_cell_lines)).ravel(),
                                'Initial': rmse_init.ravel(),
                               'Final': rmse_final.ravel()})

numpred_df = pd.DataFrame({'Cell line': np.repeat(sublinear_cell_lines, len(ct_arr)*n_reps),
                               'N_CT': np.array([np.repeat(ct_arr, n_reps)]*len(sublinear_cell_lines)).ravel(),
                                'Initial': numpred_init.ravel(),
                               'Final': numpred_final.ravel()})

param = reward
f, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 5))
sns.boxplot(data=diff_df, x='Cell line', y='Del_NumPred',
                    hue='N_CT', palette='viridis', fliersize=0, ax=ax[0])
ax[0].tick_params(axis='x', labelrotation=70, labelsize=12)
ax[0].set_ylabel(r'$\Delta$ NumPred')
ax[0].get_legend().remove()#(ncols=4)
ax[0].spines.top.set_visible(False)
ax[0].spines.right.set_visible(False)


# f, ax = plt.subplots(2, 1, sharex=True, figsize=(16, 6))
sns.boxplot(data=diff_df, x='Cell line', y='Del_RMSE',
                    hue='N_CT', palette='viridis', fliersize=0, ax=ax[1])
ax[1].tick_params(axis='x', labelrotation=70, labelsize=12)
ax[1].set_ylabel(r'$\Delta$ RMSE')
ax[1].legend(ncols=4)
ax[1].spines.top.set_visible(False)
ax[1].spines.right.set_visible(False)
f.suptitle('Reward = Penalty = '+str(param))
f.tight_layout()

f, ax = plt.subplots(2, 1, sharex=True, figsize=(5, 8))
sns.kdeplot(data=rmse_df, x='Initial', hue='N_CT',
            palette='viridis', fill=True, ax=ax[0])
ax[0].set_title('Initial RMSE')
ax[0].set_xlabel('')

sns.kdeplot(data=rmse_df, x='Final', hue='N_CT',
            palette='viridis', fill=True, ax=ax[1])
ax[1].set_title('Final RMSE')
ax[1].set_xlabel('')
f.suptitle('Reward = Penalty = '+str(param))
f.tight_layout()


f, ax = plt.subplots(2, 1, sharex=True, figsize=(5, 8))
sns.kdeplot(data=numpred_df, x='Initial', hue='N_CT',
            palette='viridis', fill=True, ax=ax[0])
ax[0].set_title('Initial NumPred')
ax[0].set_xlabel('')

sns.kdeplot(data=numpred_df, x='Final', hue='N_CT',
            palette='viridis', fill=True, ax=ax[1])
ax[1].set_title('Final NumPred')
ax[1].set_xlabel('')
f.suptitle('Reward = Penalty = '+str(param))
f.tight_layout()

diff_copy = diff_df.groupby(['Cell line', 'N_CT'])[['Del_NumPred', 'Del_RMSE']].mean().reset_index()

g = sns.lmplot(data=diff_copy, x='Del_NumPred', y='Del_RMSE', markers=['o', 's', '^', 'v'],
            hue='N_CT', palette='crest',
            scatter_kws={'edgecolor': 'face', 'alpha': 0.75, 's': 35})
g.set(ylim=g.ax.get_ylim()[::-1])
g.set_axis_labels(y_var=r'$\Delta$ RMSE', x_var=r'$\Delta$ NumPred')
plt.title('Reward = Penalty = '+str(reward))

# %%

numpred_long = numpred_df.melt(id_vars=['Cell line', 'N_CT'], value_vars=['Initial', 'Final'], value_name='NumPred', var_name='NetState')
rmse_long = rmse_df.melt(id_vars=['Cell line', 'N_CT'], value_vars=['Initial', 'Final'], value_name='RMSE', var_name='NetState')

# diff_df.loc[:, 'Rep'] = np.repeat(np.arange(1, 101)[np.newaxis, :], 124, axis=0).ravel()