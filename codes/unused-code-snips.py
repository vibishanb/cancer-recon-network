# %%
### All unused code
####### Run the model over the initialised parameters with the pickled initial network
# """No network optimisation simulations"""
# # f_count = 0
# # k = 3 # Number of cell types
# net_state = 'stat-net/'

# metabolome_pred = [[[]]]
# metabolome_measured = [[[]]]

# for i, f in enumerate(f_arr[:1]):
#     pred_temp = [[]]
#     measured_temp = [[]]
#     for j, net in enumerate(all_networks):
#         net_corrected, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(net)
#         # f_count += 1
#         ec_corr[i, j], ct_full[i, j], mean_error[i, j], pred, measured = run_network_model(f, diet, cell_line_names[j], k, cellnum_init_all[j], cellnum_final_all[j], net_corrected, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
#         pred_temp.append(pred)
#         measured_temp.append(measured)
    
#     metabolome_pred.append(pred_temp[1:])
#     metabolome_measured.append(measured_temp[1:])

# metabolome_pred = metabolome_pred[1:]
# metabolome_measured = metabolome_measured[1:]

# # %%
# ######### All plots
# figsave_flag = False
# disp_flag = True

# for i, f in enumerate(f_arr):
#     plot_all_corrs(net_state, fig_name, k, f, metabolome_pred[i], metabolome_measured[i], ec_metabolome, figsave_flag, disp_flag)

# plot_summary_stats(net_state, fig_name, k, f_arr, ec_corr, ct_full, mean_error, slopes, intercepts, figsave_flag, disp_flag)

# ######## Run model with random initial networks
# # f_count = 0
# ec_corr = np.zeros((len(f_arr), len(cell_line_names)))
# ct_full = np.zeros((len(f_arr), len(cell_line_names))) # For one cell type
# mean_error = np.zeros((len(f_arr), len(cell_line_names)))
# # ct_full = np.zeros((len(f_arr), len(cell_line_names), k)) # For more than one cell type
# # metabolome_pred = np.zeros((len(f_arr), len(cell_line_names), len(ec_metabolome)))
# metabolome_pred = [[[]]]
# metabolome_measured = [[[]]]
# slopes = np.zeros_like(ec_corr)
# intercepts = np.zeros_like(ec_corr)

# # k = 3 # Number of cell types
# net_state = 'random-net/'

# for i, f in enumerate(f_arr):
#     pred_temp = [[]]
#     measured_temp = [[]]
#     for j, rnet in enumerate(all_random_networks):
#         rnet_corrected, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(rnet)
#         ec_corr[i, j], ct_full[i, j], mean_error[i, j], pred, measured, = run_network_model(f, diet, cell_line_names[j], k, cellnum_init_all[j], cellnum_final_all[j], rnet_corrected, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
#         pred_temp.append(pred)
#         measured_temp.append(measured)

#     metabolome_pred.append(pred_temp[1:])
#     metabolome_measured.append(measured_temp[1:])

# metabolome_pred = metabolome_pred[1:]
# metabolome_measured = metabolome_measured[1:]

# # %% 
# ########### All plots
# figsave_flag = True
# disp_flag = False

# for i, f in enumerate(f_arr):
#     plot_all_corrs(net_state, fig_name, k, f, metabolome_pred[i], metabolome_measured[i], ec_metabolome, figsave_flag, disp_flag)

# plot_summary_stats(net_state, fig_name, k, f_arr, ec_corr, ct_full, mean_error, slopes, intercepts, figsave_flag, disp_flag)


# con_add_mean = np.array(consumption_added[1:]).mean(0)
# pro_add_mean = np.array(production_added[1:]).mean(0)
# con_del_mean = np.array(consumption_deleted[1:]).mean(0)
# pro_del_mean = np.array(production_deleted[1:]).mean(0)

# con_add_sd = np.array(consumption_added[1:]).std(0)
# pro_add_sd = np.array(production_added[1:]).std(0)
# con_del_sd = np.array(consumption_deleted[1:]).std(0)
# pro_del_sd = np.array(production_deleted[1:]).std(0)

# df_added = pd.DataFrame(np.array([con_add_mean, pro_add_mean]).T, columns=['uptake', 'secretion'])
# df_added.loc[:, 'metabolite'] = np.arange(net_raw.shape[0])
# df_added_long = df_added.melt(value_name='mean', value_vars=['uptake', 'secretion'], id_vars='metabolite', var_name='linkType')
# df_added_long.loc[:, 'SD'] = np.concatenate([con_add_sd, pro_add_sd])

# df_deleted = pd.DataFrame(np.array([con_del_mean, pro_del_mean]).T, columns=['uptake', 'secretion'])
# df_deleted.loc[:, 'metabolite'] = np.arange(net_raw.shape[0])
# df_deleted_long = df_deleted.melt(value_name='count', value_vars=['uptake', 'secretion'], id_vars='metabolite', var_name='linkType')
# df_deleted_long.loc[:, 'SD'] = np.concatenate([con_del_sd, pro_del_sd])

# ##### A visualisation of what is being added and removed on average, over 100 replicate runs
# met_labels = all_networks[0].loc[:, 'Metabolite'].values
# max_val = np.concatenate([df_con_plot['linkNumber'].values, df_pro_plot['linkNumber'].values]).max()

# f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(16, 18))
# sns.heatmap(data=df_con_links.iloc[:, -2:], cmap='crest', ax=ax[0],
#             yticklabels=met_labels,
#             vmin=0, vmax=max_val, cbar=False)
# ax[0].set_title('Consumption')
# ax[0].set_ylabel('')

# # sns.barplot(data=df_con_plot, x='metabolite', y='linkNumber', hue='linkType', palette='crest', ax=ax[0])
# # ax[0].set_title('Consumption links changed')
# # ax[0].set_ylabel('')

# ### Heatmap of all changes
# sns.heatmap(data=df_pro_links.iloc[:, -2:], cmap='crest', ax=ax[1],
#             yticklabels=met_labels,
#             vmin=0, vmax=max_val)
# ax[1].set_title('Production')
# ax[1].set_ylabel('')

# # f.supylabel('Changes per %d replicates' % n_reps)
# f.supxlabel('Changes per %d replicates' % n_reps)
# f.tight_layout()




# # Extract bar coordinates for error bar placement
# x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
# y_coords = [p.get_height() for p in ax.patches]

# # Adding custom error bars
# plt.errorbar(x=x_coords, y=y_coords, yerr=con_add_sd, fmt='none', c='black', capsize=3)

# f.savefig(fig_path+'/added-secretion-links.png', dpi=300)
