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
pickle_path = '../raw-output/1-celltypes/optim-net/A549-ATCC'
reward = 0.1
penalty = 0.1

[x_ori_list, x_optim_list, error_plot_list, log_bias_list, n_pred_list,
             metabolome_pred_before_list, metabolome_meas_before_list,
             metabolome_pred_after_list, metabolome_meas_after_list,
             valid_index_before_list, valid_index_after_list] = pd.read_pickle(pickle_path + '/reward-'+str(reward)+'-penalty-'+str(penalty)+'-optimised_network_output.pickle')
n_reps = len(n_pred_list)


# %%
fig = plt.figure(figsize=(14, 10))
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
ax2.spines.top.set_visible(False)
ax2.spines.right.set_visible(False)

#### How many steps of add/remove on average
sim_length = np.zeros(n_reps)
for i in range(n_reps):
    sim_length[i] = len(error_plot_list[i])
sns.histplot(sim_length, fill=True, element='step', 
             stat='proportion', alpha=0.25, bins=10, kde=True, ax=ax3)
ax3.set_xlabel('# add/remove steps')
ax3.spines.top.set_visible(False)
ax3.spines.right.set_visible(False)
# ax3.spines.left.set_visible(False)
# ax3.set_title("Step number distribution")

#### More non-trivially predicted metabolites means more error?
num_pred = np.array([arr[-1] for arr in n_pred_list])
final_error = np.array([arr[-1] for arr in log_bias_list])
sns.regplot(x=num_pred, y=final_error, color='k', ax = ax4,
            line_kws={'color': 'r', 'linewidth': 2}, scatter_kws={'s': 10})
rho, pval = spearmanr(num_pred, final_error)
ax4.text(0.3, 0.05, f'Spearman\'s $\\rho$ = {rho:.2f}', transform=ax4.transAxes) #, p-value = {pval:.2f}
# ax[1, 0].scatter(initial_error, final_error, c='k', s=5)
ax4.set_xlabel('# metabolites predicted')
ax4.set_ylabel('Final RMSE')
ax4.spines.top.set_visible(False)
ax4.spines.right.set_visible(False)
ax4.spines.left.set_bounds(final_error.min(), final_error.max())
ax4.spines.bottom.set_bounds(num_pred.min(), num_pred.max())

fig.suptitle('Reward = '+str(reward)+'; Penalty = '+str(penalty))
fig.tight_layout()

figsave_flag = True
fig_path = '../figures/1-celltypes/optim-net/A549-ATCC'
if figsave_flag:
    try:
        os.makedirs(fig_path)
    except:
        pass
    fig.savefig(fig_path+'/reward-'+str(reward)+'-penalty-'+str(penalty)+'network-predictions.png', dpi=300)
    plt.close(fig)

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

figsave_flag=True
if figsave_flag:
    fig.savefig(fig_path+'/reward-'+str(reward)+'-penalty'+str(penalty)+'prediction-comparison.png', dpi=300)
    plt.close(fig)
else:
    plt.show()


# %%
######### Figure 2
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
#### Data import
home_dir = os.getcwd()
slope_df, sublinear_cell_lines = pd.read_pickle(home_dir + '/cancer_power_law_stats.pickle')

vector_arr = []
angle_arr = []
magnitude_arr = []
for cl in sublinear_cell_lines:
    pickle_path = '../raw-output/1-celltypes/optim-net/'+cl
    reward = 0.5
    penalty = 0.5

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
    vector_arr.append(v)
    angle_arr.append(np.arctan2(v[1], v[0]))
    magnitude_arr.append(np.linalg.norm(v))

vector_arr = np.array(vector_arr)
angle_arr = np.array(angle_arr)
magnitude_arr = np.array(magnitude_arr)

heatmap_df = pd.DataFrame({'Cell line': sublinear_cell_lines,
                           'Magnitude': magnitude_arr,
                           'Direction': angle_arr,
                           'X': vector_arr[:, 0],
                           'Y': vector_arr[:, 1]})

# fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16, 5))
# sns.barplot(data=heatmap_df, x='Cell line', y='Magnitude', ax=ax[0])
# ax[0].spines.top.set_visible(False)
# ax[0].spines.right.set_visible(False)

# sns.barplot(data=heatmap_df, x='Cell line', y='Direction', ax=ax[1])
# ax[1].spines.top.set_visible(False)
# ax[1].spines.right.set_visible(False)
# ax[1].tick_params(axis='x', labelrotation=70, labelsize=12.5)
# fig.tight_layout()

# vector_df = heatmap_df.melt(id_vars=['Cell line'], value_vars=['X', 'Y'],
#                             value_name='Value', var_name='Component')

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16, 5))
sns.barplot(data=heatmap_df, x='Cell line', y='X', ax=ax[0], color='tab:red', alpha=0.9)
ax[0].spines.top.set_visible(False)
ax[0].spines.right.set_visible(False)

sns.barplot(data=heatmap_df, x='Cell line', y='Y', ax=ax[1], color='tab:red', alpha=0.9)
ax[1].spines.top.set_visible(False)
ax[1].spines.right.set_visible(False)
ax[1].tick_params(axis='x', labelrotation=70, labelsize=12.5)

fig.suptitle('Vector components; reward='+str(reward)+'; penalty='+str(penalty))
fig.tight_layout()
# %%
