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

"""
Figure 1 has been generated using a self-contained script called cancer-power-law.py that includes all the data filtering, calculations and plotting.
"""
# %%
#### Figure 2: Balanced production in one vs two celltypes, without learning
#### Data import
# cl='A549-ATCC'
# n_ct = 2
for n_ct in tqdm(np.arange(2, 7), desc='N_CT: '):
    for cl in tqdm(sublinear_cell_lines[sublinear_cell_lines != 'OVCAR-3'], desc='Cell line: '):
        pickle_path = '../raw-output/'+str(n_ct)+'-celltypes/no-learn-balanced-net/'+cl
        # reward = 0.1
        # penalty = 0.1

        # [x_ori_list, x_optim_list, error_plot_list, log_bias_list, log_bias_combined_list, n_pred_list, residual_list, balance_flag_list,
        #              metabolome_pred_before_list, metabolome_meas_before_list,
        #              metabolome_pred_after_list, metabolome_meas_after_list,
        #              valid_index_before_list, valid_index_after_list] = pd.read_pickle(pickle_path + '/reward-'+str(reward)+'-penalty-'+str(penalty)+'-optimised_network_output.pickle')

        [balance_arr_1ct, balance_arr_nct, balance_flag_arr, balanced_networks_list,
                            rmse_arr_1ct, rmse_arr_nct,
                            ec_pred_arr_1ct, ec_pred_arr_nct,
                            index_arr_1ct, index_arr_nct,
                            production_ct_arr_1ct, production_ct_arr_nct] = pd.read_pickle(pickle_path + '/balanced-networks.pickle')
        n_reps = len(balance_flag_arr)
        n_rand = 100
        npro = 115

        home_dir = os.getcwd()
        celltype_ID, celltypefreq, ec_metabolome_ID, ec_metabolome, met_baseline, core_mean = pd.read_pickle(home_dir + '/1-cells-data.pickle')
        slope_df, sublinear_cell_lines = pd.read_pickle(home_dir + '/cancer_power_law_stats.pickle')
        i_sublinear = np.where(np.isin(ec_metabolome.columns.values, sublinear_cell_lines))[0]
        ec_metabolome = ec_metabolome.iloc[:, i_sublinear]
        ec_real = ec_metabolome.loc[:, cl].values


        # max_links = int(len(x_ori_list[0].flatten())/2)

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



        # prod_df = pd.DataFrame({'Celltype': np.array([np.ones_like(rmse_arr_nct), np.ones_like(rmse_arr_nct)+1]).ravel().astype(int),
        #                    'X_pro': np.concatenate([balance_arr_nct[:, 0], balance_arr_nct[:, 1]]),
        #                    'RMSE': np.concatenate([rmse_arr_nct, rmse_arr_nct])})


        # q = sns.lineplot(data=prod_df, x='X_pro', y='RMSE',
        #                     hue='Celltype', palette=['b', 'g'], lw=3)
        # q.set_xscale('log')
        # q.set_xlabel(r'$\chi_{production}$', fontsize=17)
        # q.set_ylabel(r'RMSE')


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
                                np.where(production_ct_arr_1ct[i_min_diff]==3, 'm',
                                            np.where(production_ct_arr_1ct[i_min_diff]==4, 'saddlebrown',
                                                    np.where(production_ct_arr_1ct[i_min_diff]==5, 'cyan', 'teal')))))
        labels = np.where(production_ct_arr_1ct[i_min_diff]==1, 'CT1', 
                        np.where(production_ct_arr_1ct[i_min_diff]==2, 'CT2',
                                np.where(production_ct_arr_1ct[i_min_diff]==3, 'CT3',
                                            np.where(production_ct_arr_1ct[i_min_diff]==4, 'CT4',
                                                    np.where(production_ct_arr_1ct[i_min_diff]==5, 'CT5', 'CT6')))))

        ax[0].scatter(np.log10(ec_pred_arr_1ct[i_min_diff][index_arr_1ct[i_min_diff]]), 
                        np.log10(ec_real[index_arr_1ct[i_min_diff][0]]),
                    c=colors[index_arr_1ct[i_min_diff]],#'tab:blue',
                    alpha=0.6, edgecolor='face')
        ax[0].axline((-2, -2), (3, 3), c='k', ls='--')
        ax[0].set_title(r'$N_{CT} = 1$')
        ax[0].text(0.05, 0.9, f'RMSE = {rmse_arr_1ct[i_min_diff][0]:.2f}', transform=ax[0].transAxes)

        colors = np.where(production_ct_arr_nct[i_min_diff]==1, 'b', 
                        np.where(production_ct_arr_nct[i_min_diff]==2, 'g',
                                np.where(production_ct_arr_nct[i_min_diff]==3, 'm',
                                            np.where(production_ct_arr_nct[i_min_diff]==4, 'saddlebrown',
                                                    np.where(production_ct_arr_nct[i_min_diff]==5, 'cyan', 'black')))))
        labels = np.where(production_ct_arr_nct[i_min_diff]==1, 'CT1', 
                        np.where(production_ct_arr_nct[i_min_diff]==2, 'CT2',
                                np.where(production_ct_arr_nct[i_min_diff]==3, 'CT3',
                                            np.where(production_ct_arr_nct[i_min_diff]==4, 'CT4',
                                                    np.where(production_ct_arr_nct[i_min_diff]==5, 'CT5', 'CT6')))))
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
                                np.where(production_ct_arr_nct[i_max_diff]==3, 'm',
                                            np.where(production_ct_arr_nct[i_max_diff]==4, 'saddlebrown',
                                                    np.where(production_ct_arr_nct[i_max_diff]==5, 'cyan', 'black')))))
        labels = np.where(production_ct_arr_nct[i_max_diff]==1, 'CT1', 
                        np.where(production_ct_arr_nct[i_max_diff]==2, 'CT2',
                                np.where(production_ct_arr_nct[i_max_diff]==3, 'CT3',
                                            np.where(production_ct_arr_nct[i_max_diff]==4, 'CT4',
                                                    np.where(production_ct_arr_nct[i_max_diff]==5, 'CT5', 'CT6')))))
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

# %%
figsave_flag = 1
rmse_arr_heatmap = [[]]
num_final_heatmap = [[]]

for cl in sublinear_cell_lines:
    rmse = []
    num_final = []
    for n_ct in range(2, 7):
        # cl='A549-ATCC'
        pickle_path = '../raw-output/'+str(n_ct)+'-celltypes/no-learn-balanced-net/'+cl

        [balance_arr_1ct, balance_arr_nct, balance_flag_arr, balanced_networks_list,
                            rmse_arr_1ct, rmse_arr_nct,
                            ec_pred_arr_1ct, ec_pred_arr_nct,
                            index_arr_1ct, index_arr_nct,
                            production_ct_arr_1ct, production_ct_arr_nct] = pd.read_pickle(pickle_path + '/balanced-networks.pickle')
        rmse.append(rmse_arr_nct[balance_flag_arr].mean())
        num_final.append(balance_flag_arr.sum())
    rmse_arr_heatmap.append(rmse)
    num_final_heatmap.append(num_final)

rmse_arr_heatmap = np.array(rmse_arr_heatmap[1:])
num_final_heatmap = np.array(num_final_heatmap[1:])

heatmap_df = pd.DataFrame(rmse_arr_heatmap,
                          columns=np.arange(2, 7),
                          index=sublinear_cell_lines)

f, ax = plt.subplots(1, 1, figsize=(7.5, 15))
ax = sns.heatmap(data=heatmap_df, cmap='crest',
            annot=num_final_heatmap.tolist())
ax.set_xlabel(r'$N_{CT}$')
ax.set_ylabel('Cell line')
ax.set_title(r'RMSE vs $N_{CT}$')


if figsave_flag:
    f.figure.savefig('../figures/no-learn-rmse-vs-nct-heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(f.figure)
else:
    plt.show()

with sns.axes_style("darkgrid"):
    g = sns.boxplot(data=heatmap_df, palette='crest')
    g.set_xlabel(r'$N_{CT}$')
    g.set_ylabel('RMSE')
    g.set_title('Random networks w/o learning', pad=12)
    g.spines.top.set_visible(False)
    g.spines.right.set_visible(False)
if figsave_flag:
    g.figure.savefig('../figures/no-learn-rmse-vs-nct-boxplot.png', dpi=300, bbox_inches='tight')
    plt.close(g.figure)
else:
    plt.show()


# %%
##### Figures in slides 3-6 of network-figure-plan.pptx
fig = plt.figure(figsize=(10, 7))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])

#### A quick glance of where sims have begun and ended
colors = np.where(balance_flag_list, 'b', 'tab:red')
for i in range(n_reps):
    sns.lineplot(log_bias_list[i], ax=ax1, color=colors[i])
ax1.set_xlabel("Add/remove steps")
ax1.set_ylabel("RMSE")
ax1.set_title("%d replicate runs" % n_reps)


final_error = np.array([arr[-1] for arr in log_bias_list])
final_error_combined = np.array([arr[-1] for arr in log_bias_combined_list])
df = pd.DataFrame({'N_CT': np.concatenate([np.ones_like(final_error), np.ones_like(final_error)+1]).astype(int),
                    'Replicate': np.concatenate([np.arange(1, 1+n_reps), np.arange(1, 1+n_reps)]),
                    'Balance': np.concatenate([balance_flag_list, balance_flag_list]),
                    'RMSE': np.concatenate([final_error_combined, final_error])})
sns.lineplot(data=df, x='N_CT', y='RMSE',
            units='Replicate',
            hue='Balance', palette='coolwarm_r', hue_norm=(0, 1),
            dashes=False, estimator=None, ax=ax2, legend=True)
sns.scatterplot(data=df, x='N_CT', y='RMSE',
                hue='Balance', palette='coolwarm_r', hue_norm=(0, 1),
                edgecolor='face', alpha=0.9, ax=ax2, legend=False)
ax2.set_xlabel(r'$N_{CT}$')
ax2.set(xlim=(0.5, 2.5), xticks=[1, 2])
# ax2.get_legend().remove()

#### More non-trivially predicted metabolites means more error?
num_pred = np.array([arr[-1] for arr in n_pred_list])
final_error = np.array([arr[-1] for arr in log_bias_list])
sns.regplot(x=num_pred, y=final_error, color='k', ax = ax3,
            line_kws={'color': 'r', 'linewidth': 2}, scatter_kws={'s': 10})
rsq, pval = spearmanr(num_pred, final_error)
ax3.text(0.05, 0.9, f'rho = {rsq**2:.2f}, $p$ = {pval:.2f}', transform=ax3.transAxes)
ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax[1, 0].scatter(initial_error, final_error, c='k', s=5)
ax3.set_xlabel('# metabolites predicted')
ax3.set_ylabel('Final RMSE')

# for i in range(n_reps):
#     sns.lineplot(log_bias_list[i], ax=ax[1, 1])
# ax[1, 1].set_xlabel("Add/remove steps")
# ax[1, 1].set_ylabel("Met_RMSE")
# ax[1, 1].set_title("%d replicate runs" % n_reps)
# sns.regplot(x=initial_error, y=sim_length, color='k', ax = ax[1, 1],
#             line_kws={'color': 'r', 'linewidth': 2}, scatter_kws={'s': 10})
# rsq, pval = pearsonr(initial_error, sim_length)
# ax[1, 1].text(0.05, 0.9, f'$r^2$ = {rsq**2:.2f}, $p$ = {pval:.2f}', transform=ax[1, 1].transAxes)
# # ax[1, 0].scatter(initial_error, final_error, c='k', s=5)
# ax[1, 1].set_xlabel('Initial prediction error')
# ax[1, 1].set_ylabel('Step number')

fig.suptitle(f'Network optimisation for {cl} with reward {reward}')
fig.tight_layout()

figsave_flag = 1
fig_path = '../figures/2-celltypes/optim-net/A549-ATCC'
if figsave_flag:
    try:
        os.makedirs(fig_path)
    except:
        pass
    fig.savefig(fig_path+'/reward-'+str(reward)+'-penalty-'+str(penalty)+'-network-predictions.png', dpi=300)
    plt.close(fig)

##### Check out the best performing network
# pred_error_change = np.array([list[0]-list[-1] for list in log_bias_list])
init_pred_error = np.array([arr[0] for arr in log_bias_list])
final_pred_error = np.array([arr[-1] for arr in log_bias_list])

init_residual = np.array([arr[0] for arr in residual_list])
final_residual = np.array([arr[-1] for arr in residual_list])

i_best_net = np.where(final_pred_error == final_pred_error.min())[0] #np.where(pred_error_change == pred_error_change.max())[0]
x_ori_best = x_ori_list[i_best_net].flatten()
x_optim_best = x_optim_list[i_best_net].flatten()

p_arr_ori = np.array([i*j for i, j in zip(x_ori_best[max_links:].reshape(2, -1), [1, 2])]).sum(0)
p_arr_optim = np.array([i*j for i, j in zip(x_optim_best[max_links:].reshape(2, -1), [1, 2])]).sum(0)

met_pred_before = metabolome_pred_before_list[i_best_net][0].astype(np.float64)
met_pred_after = metabolome_pred_after_list[i_best_net][0].astype(np.float64)

met_meas_before = metabolome_meas_before_list[i_best_net][0].astype(np.float64)
met_meas_after = metabolome_meas_after_list[i_best_net][0].astype(np.float64)

i_before = valid_index_before_list[i_best_net][0].astype(bool)
i_after = valid_index_after_list[i_best_net][0].astype(bool)

c_before = np.where(p_arr_ori == 1, 'b', np.where(p_arr_ori == 2, 'g', 'tab:red'))#np.where(i_before, 'b', 'k')
c_before = np.where(i_before, c_before, 'tab:gray')

c_after = np.where(p_arr_optim == 1, 'b', np.where(p_arr_optim == 2, 'g', 'tab:red'))#np.where(i_before, 'b', 'k')
c_after = np.where(i_after, c_after, 'tab:gray')

a_before = np.where(i_before, 0.8, 0.25)
a_after = np.where(i_after, 0.8, 0.25)

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(7, 4))
ax[0].scatter(np.log10(met_pred_before), np.log10(met_meas_before), c=c_before, alpha=a_before, s=50)
# ax[0].scatter(np.log10(metabolome_pred_old+1e-7), np.log10(metabolome_measured_old+1e-7), c='k', s=9)
ax[0].axline((-2, -2), (3, 3), c='k')
ax[0].set_title('Original network')
ax[0].text(0.05, 0.8, r'$\chi_{excess}=$'+f'{init_residual[i_best_net][0].round(decimals=3):.2f}', transform=ax[0].transAxes)
# ax[0].set_xlabel(r'$log_{10}\ Predicted\ metabolome$')
ax[0].set_ylabel(r'$log_{10}\ Empirical\ data$')
ax[0].text(0.05, 0.9, f'RMSE = {init_pred_error[i_best_net][0].round(decimals=3):.2f}', transform=ax[0].transAxes)

ax[1].scatter(np.log10(met_pred_after), np.log10(met_meas_after), c=c_after, alpha=a_after, s=50)
# ax[1].scatter(np.log10(metabolome_pred+1e-7), np.log10(metabolome_measured+1e-7), c='k', s=9)
ax[1].axline((-2, -2), (2, 2), c='k')
ax[1].set_title('Optimised network')
ax[1].text(0.05, 0.9, f'RMSE = {final_pred_error[i_best_net][0].round(decimals=3):.2f}', transform=ax[1].transAxes)
ax[1].text(0.05, 0.8, r'$\chi_{excess}=$'+f'{final_residual[i_best_net][0].round(decimals=3):.2f}', transform=ax[1].transAxes)
# ax[1].set_xlabel(r'$log_{10}\ Predicted\ metabolome$')
# ax[1].set_ylabel(r'$log_{10}\ Empirical\ data$')
# ax[1].set_xlabel('', labelpad=0.1)
fig.supxlabel(r'$log_{10}\ Predicted\ metabolome$')
fig.tight_layout(pad=0.2)

figsave_flag = 1
if figsave_flag:
    fig.savefig(fig_path+'/reward-'+str(reward)+'-penalty-'+str(penalty)+'-prediction-comparison.png', dpi=300)
    plt.close(fig)
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