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
from itertools import compress

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

# Harmonized plotting style for publication-quality figures
# Use seaborn theme for consistent axes/spine/grid styles and a colorblind-friendly palette
sns.set_theme(style="ticks", context="talk", rc={"axes.grid": False})

# Fixed, reproducible, colorblind-friendly mapping for CTs (up to 6) using the
# Okabe-Ito palette (commonly used for colorblind-safe categorical plots).
# Mapping is CT index -> hex color (so colors are reproducible across runs).
CT_COLOR_MAP = {
    1: "#0072B2",  # blue
    2: "#009E73",  # bluish green
    3: "#D55E00",  # vermillion
    4: "#CC79A7",  # reddish purple / pink
    5: "#E69F00",  # orange
    6: "#56B4E9",  # sky blue
}

def ct_colors(arr):
    """Map an array of integer CT ids to hex color strings from CT_COLOR_MAP."""
    return [CT_COLOR_MAP.get(int(v), "#999999") for v in np.atleast_1d(arr)]

# Consistent scatter kwargs to avoid edgecolor warnings and ensure publication-style markers
SCATTER_KW = dict(alpha=0.8, s=45, edgecolors='none')

def add_ct_legend(ax, n_ct, title=r'$N_{CT}$'):
    """Add a small reproducible color legend for CT indices 1..n_ct positioned just outside
    the right side of the given Axes.

    The legend is anchored in axes coordinates (bbox_transform=ax.transAxes) so placing it at
    (1.02, 0.5) will put it just to the right of the axis center. Using bbox_transform avoids
    depending on figure-relative coordinates and gives consistent placement across figure sizes.
    """
    from matplotlib.patches import Patch
    # Create handles for CTs
    handles = [Patch(color=CT_COLOR_MAP[i+1], label=str(i+1)) for i in range(min(n_ct, len(CT_COLOR_MAP)))]
    # place legend just outside the right edge of the axes
    legend = ax.legend(handles=handles, title=title, bbox_to_anchor=(1.02, 0.5), loc='center left',
                       bbox_transform=ax.transAxes, frameon=False,
                       fontsize=SMALL_SIZE-3, title_fontsize=MEDIUM_SIZE-3, borderaxespad=0.0)
    # Ensure the legend is drawn on top
    legend.set_zorder(10)
    return legend


def add_ct_legend_fig(fig, n_ct, title=r'$N_{CT}$'):
    """Add a figure-level CT legend positioned just outside the right side of the figure.

    Uses fig.legend anchored in figure coordinates so it sits to the right of the entire figure
    (good for multi-panel figures). Placing at x=0.99 keeps it centered vertically while moving
    it closer to the rightmost panel compared to the previous x=1.02.
    """
    from matplotlib.patches import Patch
    handles = [Patch(color=CT_COLOR_MAP[i+1], label=str(i+1)) for i in range(min(n_ct, len(CT_COLOR_MAP)))]
    # Slightly closer to the panels (x=0.99) and centered vertically (y=0.5)
    leg = fig.legend(handles=handles, title=title, bbox_to_anchor=(0.99, 0.6), loc='center left',
                     bbox_transform=fig.transFigure, frameon=False, fontsize=SMALL_SIZE-3, title_fontsize=MEDIUM_SIZE-3)
    leg.set_zorder(10)
    return leg

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
figsave_flag = 1
#### Figure 2: Balanced production in one vs two celltypes, without learning
#### Data import
# cl='A549-ATCC'
# n_ct = 2
for n_ct in tqdm(np.arange(6, 7), desc='N_CT: '):
    for cl in tqdm(ec_metabolome.columns.to_numpy()[:-2], desc='Cell line: '):

        pickle_path = '../raw-output/'+str(n_ct)+'-celltypes/no-learn-balanced-net/'+cl

        [balance_arr_1ct, balance_arr_nct, balance_flag_arr, balanced_networks_list,
                            rmse_arr_1ct, rmse_arr_nct,
                            ec_pred_arr_1ct, ec_pred_arr_nct,
                            index_arr_1ct, index_arr_nct,
                            prod_rates_1ct, prod_rates_nct,
                            production_ct_arr_1ct, production_ct_arr_nct] = pd.read_pickle(pickle_path + '/balanced-networks.pickle')
        n_reps = len(balance_flag_arr)
        n_rand = 100
        npro = len(ec_metabolome)
        ec_real = ec_metabolome.loc[:, cl].values

        net_state = 'no-learn-balanced-net/'
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
                            edgecolors='none', alpha=0.8, zorder=2, legend=False)
        g.set_xlabel(r'$N_{CT}$')
        g.set(xlim=(0.5, n_ct+0.5), xticks=[1, n_ct])

        if figsave_flag:
            g.figure.savefig(fig_path+'/balance-pairwise-comparison-npro-'+str(npro)+'.png', dpi=300, bbox_inches='tight')
            plt.close(g.figure)
        else:
            plt.show()

        # Use a saturated categorical palette for N_CT to be more distinct than 'crest'
        palette_nct = sns.color_palette('tab10', n_colors=n_ct)
        h = sns.boxplot(data=df, x='Balance', y='RMSE',
                        hue='N_CT', palette=palette_nct)
        h.set_xlabel('Balance state')
        h.set(xlim=(-0.75, 1.75), xticks=[0, 1], xticklabels=['False', 'True'])
        h.get_legend().set_title(r'$N_{CT}$')
        if figsave_flag:
            h.figure.savefig(fig_path+'/balance-comparison-boxplot-npro-'+str(npro)+'.png', dpi=300, bbox_inches='tight')
            plt.close(h.figure)
        else:
            plt.show()


        # if n_ct == 2:
        #     prod_df = pd.DataFrame({'Xpro_CT1': balance_arr_nct[:, 0],
        #                             'Xpro_CT2': balance_arr_nct[:, 1],
        #                             'RMSE': rmse_arr_nct})
        #     q = sns.scatterplot(data=prod_df, x='Xpro_CT1', y='Xpro_CT2',
        #                         hue = 'RMSE', palette='flare',#sizes=(50, 200),
        #                 # hue='N_produced', palette='crest',
        #                 edgecolor= 'face', alpha = 0.8, s=75)
        #     q.set_xscale('log')
        #     q.set_yscale('log')
        #     q.set_xlabel(r'$\chi_{production, CT1}$')
        #     q.set_ylabel(r'$\chi_{production, CT2}$')
        #     q.axvspan(xmin=0.1, xmax=1, alpha=0.2, color='tab:green')
        #     q.axhspan(ymin=0.1, ymax=1, alpha=0.2, color='tab:green')
        #     # q.axhline(y=1, linestyle='dashed', c='tab:green')
        #     # q.text(0.9, 0.38, r'$y=1$', c='tab:green', transform=g.transAxes)
        #     # q.axvline(x=1, linestyle='dashed', c='tab:green')
        #     # q.text(0.68, 0.87, r'$x=1$', c='tab:green', rotation='vertical', transform=g.transAxes)

        # else:
        prod_df = pd.DataFrame(balance_arr_nct, columns=np.arange(1, n_ct+1)).join(pd.Series(rmse_arr_nct, name='RMSE')).melt(id_vars='RMSE', value_name='Balance', var_name='N_CT')

        q = sns.scatterplot(data=prod_df, x='Balance', y='RMSE',
                            style = 'N_CT', markers=True, # palette='flare',#sizes=(50, 200),
                    # hue='N_produced', palette='crest',
                    edgecolor= 'k', alpha = 0.75, s=45)
        q.set_xscale('log')
        q.axvspan(xmin=0.1, xmax=1, alpha=0.2, color='tab:green')
        # q.set_yscale('log')
        q.set_xlabel(r'$\chi_{production}$')
        q.set_ylabel('RMSE')
        q.get_legend().set_title(r'$N_{CT}$')


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
        # Legacy explicit color mapping for CTs replaced with CT_COLOR_MAP (colorblind-friendly palette)
        # labels are simplified to CT{n} when needed
        # colors and labels will be mapped inline where scatter calls occur using CT_COLOR_MAP or ct_colors().
        labels = None

        ax[0].scatter(np.log10(ec_pred_arr_1ct[i_min_diff][index_arr_1ct[i_min_diff]]), 
                        np.log10(ec_real[index_arr_1ct[i_min_diff][0]]),
                    c=ct_colors(production_ct_arr_1ct[i_min_diff][index_arr_1ct[i_min_diff]]),
                    alpha=0.6, edgecolors='none')
        ax[0].axline((-2, -2), (3, 3), c='k', ls='--')
        ax[0].set_title(r'$N_{CT} = 1$')
        ax[0].text(0.05, 0.9, f'RMSE = {rmse_arr_1ct[i_min_diff][0]:.2f}', transform=ax[0].transAxes, fontsize=SMALL_SIZE-4, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'), zorder=5, clip_on=False)

        # Legacy explicit color mapping for CTs replaced with CT_COLOR_MAP (colorblind-friendly palette)
        labels = None

        ax[1].scatter(np.log10(ec_pred_arr_nct[i_min_diff][index_arr_nct[i_min_diff]]), 
                        np.log10(ec_real[index_arr_nct[i_min_diff][0]]),
                    c=ct_colors(production_ct_arr_nct[i_min_diff][index_arr_nct[i_min_diff]]), marker='o',
                    alpha=0.6, edgecolors='none')

        ax[1].axline((-2, -2), (3, 3), c='k', ls='--')
        ax[1].set_title(r'Worst of $N_{CT} = $'+str(n_ct))
        ax[1].text(0.5, 0.9, f'RMSE = {rmse_arr_nct[i_min_diff][0]:.2f}', transform=ax[1].transAxes, fontsize=SMALL_SIZE-4, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'), zorder=5, clip_on=False)

        #### Maximum difference
        # Legacy explicit color mapping for CTs replaced with CT_COLOR_MAP (colorblind-friendly palette)
        labels = None
        
        ax[2].scatter(np.log10(ec_pred_arr_nct[i_max_diff][index_arr_nct[i_max_diff]]), 
                        np.log10(ec_real[index_arr_nct[i_max_diff][0]]),
                    c=ct_colors(production_ct_arr_nct[i_max_diff][index_arr_nct[i_max_diff]]), marker='o',
                    alpha=0.6, edgecolors='none')

        ax[2].axline((-2, -2), (3, 3), c='k', ls='--')
        ax[2].text(0.6, 0.1, f'RMSE = {rmse_arr_nct[i_max_diff][0]:.2f}', transform=ax[2].transAxes, fontsize=SMALL_SIZE-4, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'), zorder=5, clip_on=False)
        ax[2].set_title(r'Best of $N_{CT} = $'+str(n_ct))
        # add reproducible CT legend inset for clarity
        # For multi-panel plot, add a single, shared figure-level CT legend to the right
        try:
            add_ct_legend_fig(f, n_ct)
        except Exception:
            # fall back to axis-level legend if something unexpected happens
            try:
                add_ct_legend(ax[2], n_ct)
            except Exception:
                pass

        f.supxlabel(r'$log_{10}\ Prediction$', fontsize=15)
        f.supylabel(r'$log_{10}\ Data$', fontsize=15)
        f.tight_layout()

        if figsave_flag:
            f.figure.savefig(fig_path+'/metabolome-prediction-comparison-npro-'+str(npro)+'.png', dpi=300, bbox_inches='tight')
            plt.close(f.figure)
        else:
            plt.show()

#### Pooled pair plots
rmse_pooled_1ct, rmse_pooled_nct, balance_flag_pooled, rep_num = [[]], [[]], [[]], [[]]
# slope_df, sublinear_cell_lines = pd.read_pickle(home_dir + '/cancer_power_law_stats.pickle')
for cl in ec_metabolome.columns.to_numpy()[:-2]:
    pickle_path = '../raw-output/2-celltypes/no-learn-balanced-net/'+cl

    [balance_arr_1ct, balance_arr_nct, balance_flag_arr, balanced_networks_list,
                        rmse_arr_1ct, rmse_arr_nct,
                        ec_pred_arr_1ct, ec_pred_arr_nct,
                        index_arr_1ct, index_arr_nct,
                        prod_rates_1ct, prod_rates_nct,
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
reps = len(rmse_pooled_1ct.ravel())
df_pooled = pd.DataFrame({'Replicate': np.concatenate([np.arange(1, reps+1), np.arange(1, reps+1)]).ravel(),
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
                    edgecolors='none', alpha=0.8, zorder=2, legend=False)
ax.set_xlabel(r'$N_{CT}$')
ax.set(xlim=(0.5, 2.5), xticks=[1, 2])
ax.get_legend().set_title('Balance')

net_state = 'no-learn-balanced-net/'
fig_path = '../figures/2-celltypes/'+net_state
if figsave_flag:
    g.figure.savefig(fig_path+'/balance-rmse-comparison-all-cell-lines-pooled.png', dpi=300, bbox_inches='tight')
    plt.close(g.figure)
else:
    plt.show()

# %%
figsave_flag = 1
########## Figure 3
rmse_arr_heatmap = [[]]
num_final_heatmap = [[]]

for cl in ec_metabolome.columns.to_numpy()[:-2]:
    rmse = []
    num_final = []
    ec_real = ec_metabolome.loc[:, cl].values
    for n_ct in range(2, 7):
        # cl='A549-ATCC'
        pickle_path = '../raw-output/'+str(n_ct)+'-celltypes/no-learn-balanced-net/'+cl

        [balance_arr_1ct, balance_arr_nct, balance_flag_arr, balanced_networks_list,
                            rmse_arr_1ct, rmse_arr_nct,
                            ec_pred_arr_1ct, ec_pred_arr_nct,
                            index_arr_1ct, index_arr_nct,
                            prod_rates_1ct, prod_rates_nct,
                            production_ct_arr_1ct, production_ct_arr_nct] = pd.read_pickle(pickle_path + '/balanced-networks.pickle')
        rmse.append(rmse_arr_nct[balance_flag_arr].mean())
        num_final.append(balance_flag_arr.sum())

        f, ax = plt.subplots(1, 1, figsize=(4, 3.5))

        rmse_diff = rmse_arr_1ct - rmse_arr_nct
        i_min_diff = np.where(rmse_diff == rmse_diff.min(), True, False)
        i_max_diff = np.where(rmse_diff == rmse_diff.max(), True, False)

        # Legacy explicit color mapping removed; use CT_COLOR_MAP for colors when plotting
        labels = None
        
        f = plt.scatter(np.log10(ec_pred_arr_nct[i_max_diff][index_arr_nct[i_max_diff]]), 
                                np.log10(ec_real[index_arr_nct[i_max_diff][0]]),
                            c=ct_colors(production_ct_arr_nct[i_max_diff][index_arr_nct[i_max_diff]]), marker='o',
                            alpha=0.6, edgecolors='none')

        plt.axline((-2, -2), (3, 3), c='k', ls='--')
        plt.text(0.6, 0.1, f'RMSE = {rmse_arr_nct[i_max_diff][0]:.2f}', transform=f.axes.transAxes, fontsize=SMALL_SIZE-4, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'), zorder=5, clip_on=False)
        plt.title(cl+str('; ')+r'$N_{CT} = $'+str(n_ct))
        plt.xlabel(r'$log_{10}\ Prediction$', fontsize=15)
        plt.ylabel(r'$log_{10}\ Data$', fontsize=15)
        # add CT legend inset for reproducibility
        try:
            add_ct_legend(plt.gca(), n_ct)
        except Exception:
            pass

        net_state = 'no-learn-balanced-net/'
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
                          columns=np.arange(2, 7),
                          index=ec_metabolome.columns.to_numpy()[:-2])

f, ax = plt.subplots(1, 1, figsize=(7.5, 15))
ax = sns.heatmap(data=heatmap_df, cmap='crest')
ax.set_xlabel(r'$N_{CT}$')
ax.set_ylabel('Cell line')
ax.set_title(r'RMSE vs $N_{CT}$')

if figsave_flag:
    f.figure.savefig('../figures/no-learn-rmse-vs-nct-heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(f.figure)
else:
    plt.show()

with sns.axes_style("ticks"):
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
figsave_flag = 1
##### Figure 4
for k in range(1):#tqdm(range(len(sublinear_cell_lines[0])), desc='Cell line: '):#np.arange(0, 1):
        for n_ct in np.arange(5, 6):
            reward = 0.2
            penalty = 0.2
            cl = sublinear_cell_lines[k]
            ec_real = ec_metabolome.loc[:, cl]

            pickle_path = '../raw-output/'+str(n_ct)+'-celltypes/optim-net/'+cl

            [x_all_list, x_ori_list, x_optim_list, error_plot_list, log_bias_list, log_bias_combined_list, log_bias_list_null, x_all_list_null,#x_ori_list_null, x_optim_list_null,
                         n_pred_list, residual_list, balance_flag_list, prod_overlap_list, con_overlap_list,
                        metabolome_pred_before_list, metabolome_meas_before_list,
                        metabolome_pred_after_list, metabolome_meas_after_list,
                        valid_index_before_list, valid_index_after_list] = pd.read_pickle(pickle_path + '/reward-'+str(reward)+'-penalty-'+str(penalty)+'-optimised_network_output.pickle')
            max_links = int(len(x_ori_list[0].flatten())/2)
            n_reps = len(n_pred_list)

            init_pred_error = np.array([arr[0] for arr in log_bias_list])
            final_pred_error = np.array([arr[-1] for arr in log_bias_list])

            init_residual = np.array([arr[0] for arr in residual_list])
            final_residual = np.array([arr[-1] for arr in residual_list])

            #### A glance of where sims have begun and ended
            # colors = np.where(balance_flag_list, 'b', 'tab:red')
            for i in range(n_reps):
                g = sns.lineplot(log_bias_list[i], color='b')
            g.set_xlabel('Learning steps')
            g.set_ylabel('Prediction error')
            sns.despine(offset=0, trim=False)
            g.set_title(f'Network optimisation for {cl} with reward {reward}')
            g.text(0.8, 0.9, f'n={len(log_bias_list)}', transform=g.axes.transAxes, fontsize=MEDIUM_SIZE)
            g.figure.tight_layout(pad=0.6)


            fig_path = '../figures/'+str(n_ct)+'-celltypes/optim-net/'+cl
            if figsave_flag:
                try:
                    os.makedirs(fig_path)
                except:
                    pass
                g.figure.savefig(fig_path+'/reward-'+str(reward)+'-penalty-'+str(penalty)+'-network-learning-progress.png', dpi=300)
                plt.close(g.figure)

            #### More overlap means less error?
            num_prod_overlap = []
            num_con_overlap = []
            for x in x_optim_list:
                ## Production overlap
                optim_list = x[max_links:].reshape(n_ct, -1)
                prod_overlap = np.array([optim_list[:, i].sum() for i in range(89)])
                num_prod_overlap.append(len(prod_overlap[prod_overlap > 1]))
                
                ## Consumption overlap
                optim_list = x[:max_links].reshape(n_ct, -1)
                con_overlap = np.array([optim_list[:, i].sum() for i in range(89)])
                num_con_overlap.append(len(con_overlap[con_overlap > 1]))
            num_con_overlap, num_prod_overlap = np.array(num_con_overlap), np.array(num_prod_overlap)
            
            overlap_df = pd.DataFrame({'LinkType': np.concatenate([np.repeat('Consumption', len(num_con_overlap)), np.repeat('Production', len(num_prod_overlap))]),
                                    #    'Balance': np.concatenate([balance_flag_list, balance_flag_list]),
                                       'Overlap': np.concatenate([num_con_overlap, num_prod_overlap]),
                                       'RMSE': np.concatenate([final_pred_error, final_pred_error])})
            with sns.axes_style('ticks'):
                f = sns.catplot(data=overlap_df, x='Overlap',
                                 col='LinkType', hue='LinkType', palette={'Consumption': 'tab:green', 'Production': 'tab:blue'}, kind='count', stat='percent',
                                 alpha=0.9, fill=True)
                f.set_xlabels('# overlapping metabolites')
                f.set_ylabels('Percentage')
                f.set_titles('{col_name} overlap')
                f.legend.remove()
                
                sns.despine(offset=4, trim=False)
                f.figure.tight_layout()

                # f = sns.lmplot(data=overlap_df, x='Overlap', y='RMSE', row='LinkType',
                #         #    hue='Balance', palette={False: 'r', True: 'b'},
                #            line_kws={'linewidth': 2}, scatter_kws={'s': 15},
                #            facet_kws=dict(sharex=False))
                # f.set_xlabels('')
                # f.figure.supxlabel('# overlapping metabolites', size=MEDIUM_SIZE)
                # # f.figure.tight_layout()

            if figsave_flag:
                f.figure.savefig(fig_path+'/reward-'+str(reward)+'-penalty-'+str(penalty)+'-consumption-niche-overlap-histplot.png', dpi=300)
                plt.close(f.figure)
            else:
                plt.show()            

# %%
figsave_flag = 1
MAX_ID_metabolites=89
max_links = n_ct*MAX_ID_metabolites
met_ID = diet.index.to_numpy()
linktype_error_df = pd.DataFrame()
n_reps = len(log_bias_list)

init_error = np.array([arr[0] for arr in log_bias_list])
final_error = np.array([arr[-1] for arr in log_bias_list])
error_change = init_error - final_error
i_best = np.where(error_change == error_change.max())[0] # np.where(final_error == final_error.min())[0]

pooled_df = pd.DataFrame()
for k in range(len(x_ori_list)):
    linkchange = []
    linktype = []
    changetype = []
    overlapmets = []
    test = []
    x_list = x_all_list[k]
    elist = log_bias_list[k]
    for i in range(1, len(x_list)):
        x_diff = x_list[i] - x_list[i-1]
        i_change = np.where(x_diff != 0)[0]

        # for i_change in i_change_all:
        changetype.append(np.where(x_diff[i_change] > 0, 'Added', 'Removed')[0])

        if i_change < max_links:
            linktype.append('Consumption')
            ct = i_change//MAX_ID_metabolites
            l1, m1 = calculate_overlap_stats(x_list[i-1], n_ct, max_links, met_ID)[:2]
            l2, m2 = calculate_overlap_stats(x_list[i], n_ct, max_links, met_ID)[:2]

        else:
            linktype.append('Production')
            ct = (i_change - max_links)//MAX_ID_metabolites
            l1, m1 = calculate_overlap_stats(x_list[i-1], n_ct, max_links, met_ID)[2:]
            l2, m2 = calculate_overlap_stats(x_list[i], n_ct, max_links, met_ID)[2:]

        d1 = {m: l for m, l in zip(m1, l1)}
        d2 = {m: l for m, l in zip(m2, l2)}
        lc_temp = []
        for s in (d2.items() ^ d1.items()):
            lc_temp.append(s)
            test.append(len((d2.items() ^ d1.items())))
        # lc_temp = l2.copy()
        
        # overlapmets.append(m2.copy())
        
        if len(lc_temp) > 1:
            final_lc = np.where(changetype[-1] == 'Added',
                                list(compress(lc_temp, [s in d2.items() for s in lc_temp])),
                                list(compress(lc_temp, [s in d1.items() for s in lc_temp])))
            linkchange.append([s[1] for s in final_lc])
        elif len(lc_temp) == 0:
            linkchange.append(str(ct[0]+1))        
        else:
            linkchange.append([s[1] for s in lc_temp])
            # linkchange.append(lc_temp)

    i_valid = np.array([np.where(len(l) == 0, False, True) for l in linkchange])
    df = pd.DataFrame({'LinkType': list(compress(linktype, i_valid)),
                        'ChangeType': list(compress(changetype, i_valid)),
                        'Link': [str(arr[0]) for arr in linkchange]})

    linklen = np.array([len(l) for l in df['Link'].values])
    # overlaplen = (1 + (linklen/2)).astype(int)

    df.loc[:, 'OverlapLen'] = (1 + (linklen/2)).astype(int) - 1
    # df.loc[:, 'RMSEType'] = np.where(df['RMSEdiff'].values < 0, True, False)
    df.loc[:, 'SimTime'] = np.arange(len(df))
    pooled_df = pd.concat([pooled_df, df])

pooled_df_null = pd.DataFrame()
for k in range(len(x_all_list_null)):
    linkchange = []
    linktype = []
    changetype = []
    overlapmets = []
    test = []
    x_list = x_all_list_null[k]
    elist = log_bias_list[k]
    for i in range(1, len(x_list)):
        x_diff = x_list[i] - x_list[i-1]
        i_change = np.where(x_diff != 0)[0]

        # for i_change in i_change_all:
        changetype.append(np.where(x_diff[i_change] > 0, 'Added', 'Removed')[0])

        if i_change < max_links:
            linktype.append('Consumption')
            ct = i_change//MAX_ID_metabolites
            l1, m1 = calculate_overlap_stats(x_list[i-1], n_ct, max_links, met_ID)[:2]
            l2, m2 = calculate_overlap_stats(x_list[i], n_ct, max_links, met_ID)[:2]

        else:
            linktype.append('Production')
            ct = (i_change - max_links)//MAX_ID_metabolites
            l1, m1 = calculate_overlap_stats(x_list[i-1], n_ct, max_links, met_ID)[2:]
            l2, m2 = calculate_overlap_stats(x_list[i], n_ct, max_links, met_ID)[2:]

        d1 = {m: l for m, l in zip(m1, l1)}
        d2 = {m: l for m, l in zip(m2, l2)}
        lc_temp = []
        for s in (d2.items() ^ d1.items()):
            lc_temp.append(s)
            test.append(len((d2.items() ^ d1.items())))
        # lc_temp = l2.copy()
        
        # overlapmets.append(m2.copy())
        
        if len(lc_temp) > 1:
            final_lc = np.where(changetype[-1] == 'Added',
                                list(compress(lc_temp, [s in d2.items() for s in lc_temp])),
                                list(compress(lc_temp, [s in d1.items() for s in lc_temp])))
            linkchange.append([s[1] for s in final_lc])
        elif len(lc_temp) == 0:
            linkchange.append(str(ct[0]+1))        
        else:
            linkchange.append([s[1] for s in lc_temp])
            # linkchange.append(lc_temp)

    i_valid = np.array([np.where(len(l) == 0, False, True) for l in linkchange])
    df = pd.DataFrame({'LinkType': list(compress(linktype, i_valid)),
                        'ChangeType': list(compress(changetype, i_valid)),
                        'Link': [str(arr[0]) for arr in linkchange]})

    linklen = np.array([len(l) for l in df['Link'].values])
    # overlaplen = (1 + (linklen/2)).astype(int)

    df.loc[:, 'OverlapLen'] = (1 + (linklen/2)).astype(int) - 1
    # df.loc[:, 'RMSEType'] = np.where(df['RMSEdiff'].values < 0, True, False)
    df.loc[:, 'SimTime'] = np.arange(len(df))
    pooled_df_null = pd.concat([pooled_df, df])

pooled_df_null.loc[:, 'Model'] = 'Null'
pooled_df.loc[:, 'Model'] = 'Optimised'

# %%
figsave_flag=1
fig_path = '../figures/'+str(n_ct)+'-celltypes/optim-net/'+cl

pooled_df.loc[:, 'OverlapStatus'] = np.where(pooled_df.loc[:, 'OverlapLen'] > 1, 'Overlap',
                                             np.where((pooled_df.loc[:, 'OverlapLen'] == 1)*(pooled_df.loc[:, 'LinkType'] == 'Consumption'), 'Consumption', 'Production'))

pooled_df_null.loc[:, 'OverlapStatus'] = np.where(pooled_df_null.loc[:, 'OverlapLen'] > 1, 'Overlap',
                                             np.where((pooled_df_null.loc[:, 'OverlapLen'] == 1)*(pooled_df_null.loc[:, 'LinkType'] == 'Consumption'), 'Consumption', 'Production'))

pooled_df[pooled_df.columns[-1]] = pd.Categorical(pooled_df[pooled_df.columns[-1]],
                                    ['Overlap', 'Consumption', 'Production'])
g = sns.histplot(data=pooled_df, x='OverlapStatus',
                hue='OverlapStatus', palette={'Overlap': 'tab:grey', 'Consumption': 'tab:green', 'Production': 'tab:blue'},
                stat='probability', alpha=0.8, fill=True,
                legend=False, shrink=0.7, discrete=True)
g.set(ylabel=r'Fraction of learnt networks', xlabel='Link type')
g.figure.tight_layout()
sns.despine(offset=3, trim=False)

if figsave_flag:
    g.figure.savefig(fig_path+'/reward-'+str(reward)+'-penalty-'+str(penalty)+'-overlap-status-fraction.png', dpi=300)
    plt.close(g.figure)
else:
    plt.show()
# g.set(yscale='log')

error_change_null = np.array([ierr - nerr for nerr, ierr in zip(log_bias_list_null, init_error)]).flatten()
null_rmse_df = pd.DataFrame({'Model': np.concatenate([['Null']*len(error_change_null), ['Optimised']*len(error_change)]),
                             'DeltaRMSE': np.concatenate([error_change_null, error_change])})

h = sns.kdeplot(data=null_rmse_df, x='DeltaRMSE',
                 hue='Model', palette={'Null': 'tab:grey', 'Optimised': 'tab:red'},
                 alpha=0.75, fill=True, multiple='layer', common_norm=False)

h.set_xlabel(r'$\Delta$ RMSE = Initial $-$ Final')
sns.despine(offset=4, trim=False)
h.figure.tight_layout()
if figsave_flag:
    h.figure.savefig(fig_path+'/reward-'+str(reward)+'-penalty-'+str(penalty)+'-null-vs-optimised-delta-rmse-distribution.png', dpi=300)
    plt.close(h.figure)
else:
    plt.show()

p = sns.histplot(data=pd.concat([pooled_df_null, pooled_df]), x='OverlapLen',
                 hue='Model', palette={'Null': 'tab:grey', 'Optimised': 'tab:red'},
                 alpha=0.75, fill=True, discrete=True, shrink=0.8,
                 multiple='layer', stat='proportion',
                 common_norm=False)
p.set_xlabel(r'# overlapping links')
p.xaxis.set_major_locator(MaxNLocator(integer=True))
sns.despine(offset=4, trim=False)
p.figure.tight_layout()
if figsave_flag:
    p.figure.savefig(fig_path+'/reward-'+str(reward)+'-penalty-'+str(penalty)+'-null-vs-optimised-link-length-distribution.png', dpi=300)
    plt.close(p.figure)
else:
    plt.show()




# %%
