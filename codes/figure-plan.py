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
import pickle
import os

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
#### Figure 2: Single cell type model predictions-error reduction and number of metabolites predicted non-trivially
#### Data import
pickle_path = '../raw-output/2-celltypes/optim-net/balance-partition/A549-ATCC'
reward = 0.1
penalty = 0.1

[x_ori_list, x_optim_list, error_plot_list, log_bias_list, n_pred_list, residual_list, balance_flag_list,
             metabolome_pred_before_list, metabolome_meas_before_list,
             metabolome_pred_after_list, metabolome_meas_after_list,
             valid_index_before_list, valid_index_after_list] = pd.read_pickle(pickle_path + '/reward-'+str(reward)+'-penalty-'+str(penalty)+'-optimised_network_output.pickle')
n_reps = len(n_pred_list)
max_links = int(len(x_ori_list[0].flatten())/2)


# %%
##### Figures in slides 3-6 of network-figure-plan.pptx
fig = plt.figure(figsize=(17, 10))
gs = GridSpec(2, 3, figure=fig)
gs01 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, :2])
gs02 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 2])

ax_left = gs01.subplots(sharex=True)
ax1 = ax_left[0]#fig.add_subplot(ax_left[0])
ax2 = ax_left[1]#fig.add_subplot(ax_left[1])
ax3 = fig.add_subplot(gs02[0])
ax4 = fig.add_subplot(gs02[1])

#### A quick glance of where sims have begun and ended
hue_vector = np.arange(len(n_pred_list))
for i in range(n_reps):
    sns.lineplot(log_bias_list[i], ax=ax1, linewidth=2.5,)
# ax1.set_xlabel("")
ax1.set_ylabel("RMSE")
ax1.set_title("%d replicate runs" % n_reps)
ax1.spines.top.set_visible(False)
ax1.spines.right.set_visible(False)

#### Are more non-trivial predictions made on average during optimisation?
for i in range(n_reps):
    sns.lineplot(n_pred_list[i], ax=ax2, linewidth=2.5)
ax2.set_xlabel("Add/remove steps")
ax2.set_ylabel("# metabolites predicted")
# ax2.spines.top.set_visible(False)
# ax2.spines.right.set_visible(False)


# #### How many steps of add/remove on average
# sim_length = np.zeros(n_reps)
# for i in range(n_reps):
#     sim_length[i] = len(error_plot_list[i])
# sns.histplot(sim_length, fill=True, element='step', 
#              stat='proportion', alpha=0.25, bins=10, kde=True, ax=ax3)
# ax3.set_xlabel('# add/remove steps')
# # ax3.spines.top.set_visible(False)
# # ax3.spines.right.set_visible(False)
# # ax3.spines.left.set_visible(False)
# # ax3.set_title("Step number distribution")

#### More non-trivially predicted metabolites means more error?
num_pred = np.array([arr[-1] for arr in n_pred_list])
final_error = np.array([arr[-1] for arr in log_bias_list])
sns.regplot(x=num_pred, y=final_error, color='k', ax = ax3,
            line_kws={'color': 'r', 'linewidth': 2}, scatter_kws={'s': 10})
rho, pval = spearmanr(num_pred, final_error)
ax3.text(0.3, 0.05, f'Spearman\'s $\\rho$ = {rho:.2f}', transform=ax3.transAxes) #, p-value = {pval:.2f}
# ax[1, 0].scatter(initial_error, final_error, c='k', s=5)
ax3.set_xlabel('# metabolites predicted')
ax3.set_ylabel('Final RMSE')
ax3.spines.top.set_visible(False)
ax3.spines.right.set_visible(False)
ax3.spines.left.set_bounds(final_error.min(), final_error.max())
ax3.spines.bottom.set_bounds(num_pred.min(), num_pred.max())

#### Normalised residual production fraction-how does balance change during optimisation?
log_residual_list = [np.log10(np.array(arr) + 1e-5) for arr in residual_list]
for i in range(n_reps):
    sns.lineplot(residual_list[i], ax=ax4, linewidth=2.5)
# ax4.fill_between(x=ax4.get_xlim(), y1=0, y2=-1,
#                  alpha=0.4, color='tab:green', lw=2, ls='--')
ax4.set_ylabel(r"$\chi_{excess}$")
ax4.set_xlabel('Add/remove steps')

fig.suptitle('Reward = '+str(reward)+'; Penalty = '+str(penalty))
# fig.tight_layout()

figsave_flag = False
fig_path = '../figures/2-celltypes/optim-net/A549-ATCC'
if figsave_flag:
    try:
        os.makedirs(fig_path)
    except:
        pass
    fig.savefig(fig_path+'/reward-'+str(reward)+'-penalty-'+str(penalty)+'network-predictions.png', dpi=300)
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

c_before = np.where(p_arr_ori == 1, 'tab:blue', np.where(p_arr_ori == 2, 'tab:green', 'tab:red'))#np.where(i_before, 'b', 'k')
c_before = np.where(i_before, c_before, 'tab:gray')
c_after = np.where(p_arr_ori == 1, 'tab:blue', np.where(p_arr_ori == 2, 'tab:green', 'tab:red'))#np.where(i_before, 'b', 'k')
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

figsave_flag=False
if figsave_flag:
    fig.savefig(fig_path+'/reward-'+str(reward)+'-penalty'+str(penalty)+'prediction-comparison.png', dpi=300)
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