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


############# From no-learning code ####################

# # %%
# home_dir = os.getcwd()
# n_ct = 2
# all_networks, i_intake, names = pd.read_pickle(home_dir + '/' + str(n_ct) + '-cells-cancer_network.pickle')
# # i_selfish = 0

# # pickle_in = open("data.pickle","rb")
# celltype_ID, celltypefreq, ec_metabolome_ID, ec_metabolome, met_baseline, core_mean = pd.read_pickle(home_dir + '/' + str(n_ct) + '-cells-data.pickle')
# cell_line_names = ec_metabolome.columns.to_numpy()

# ## Source for initial and final cell numbers for the various cell lines: https://www.thermofisher.com/in/en/home/references/gibco-cell-culture-basics/cell-culture-protocols/cell-culture-useful-numbers.html
# n_lines = len(core_mean.loc[:, 'Cell line'])
# # nonadh_cell_lines = ['SR', 'MOLT-4', 'HL-60(TB)', 'K562', 'RPMI 8226', 'CCRF-CEM']
# # i_nonadh = np.where(np.isin(core_mean.loc[:, 'Cell line'], nonadh_cell_lines))[0]

# t75_cell_lines = ['NCI-H460', 'HCC-2998', 'SW620']
# i_t75 = np.where(np.isin(cell_line_names, t75_cell_lines))[0]
# cellnum_init_all = np.array([4.9e+06]*n_lines)
# cellnum_init_all[i_t75] = 2.1e+06
# cellnum_final_all = np.array([23.3e+06]*n_lines)
# cellnum_final_all[i_t75] = 8.4e+06

# slope_df, sublinear_cell_lines = pd.read_pickle(home_dir + '/cancer_power_law_stats.pickle')
# nets_temp = all_networks.copy()

# ### Filtering those networks for cell lines with power law slopes
# i_sublinear = np.where(np.isin(ec_metabolome.columns.values, sublinear_cell_lines))[0]
# all_networks = []
# for i in i_sublinear:
#     all_networks.append(nets_temp[i])
# ec_metabolome = ec_metabolome.iloc[:, i_sublinear]

# i_nonzero_celltypes = all_networks[0]['celltypes_ID'].unique()
# i_nonzero_celltypes = np.sort(i_nonzero_celltypes)
# i_nonzero_celltypes = celltype_ID.values.copy()
# i_nonzero_metabolites = all_networks[0]['metabolites_ID'].unique()
# # i_nonzero_metabolites = np.sort(i_nonzero_metabolites)

# cellnum_init_all = cellnum_init_all[i_sublinear]
# cellnum_final_all = cellnum_final_all[i_sublinear]

# MAX_ID_celltypes = len(i_nonzero_celltypes)  # MAX_ID_celltypes is the maximum of ID labels for celltypes.
# MAX_ID_metabolites = len(i_nonzero_metabolites)  # MAX_ID_metabolites is the maximum of ID labels for metabolites.

# diet = met_baseline.mean(axis=1)

# ####### Generate random networks, one for each cell line
# # for i in range(len(all_networks)):
# all_random_networks = []
# for i, net in enumerate(all_networks):
#     bias_metabolome = ec_metabolome.iloc[:, i].values - diet.values
#     rnet = generate_random_network(net, bias_metabolome)
#     all_random_networks.append(rnet)

# # %%
# ###### Random production levels and celltype abundance for one cell line
# f = 0.55
# in_degree_flag = False

# net, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(all_random_networks[0])
# ec_real = ec_metabolome.iloc[:, 0].values

# rmse_arr = []
# n_produced_arr = []
# cf_arr = []
# for i in range(1, 11):
#     x = np.where(net.iloc[:, -1] > 0, 1, 0)

#     max_links = MAX_ID_metabolites * MAX_ID_celltypes

#     n_pro = i*10
#     edges = np.append(np.zeros(MAX_ID_metabolites-n_pro), np.ones(n_pro))
#     x[max_links:] = np.random.permutation(np.repeat(edges[np.newaxis, :], 2, axis=0).ravel())

#     net_mod = net_from_x(x, MAX_ID_metabolites=MAX_ID_metabolites, MAX_ID_celltypes=MAX_ID_celltypes)
#     for cf in np.linspace(0.1, 0.9, 10):
#         ct_freq = np.array([cf, 1-cf])
#         m2b, b2m, ec_pred = calculate_metabolome_from_net(f, ct_freq, diet, net_mod, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

#         i_nonzero = np.where(ec_pred * ec_real, True, False)
#         i_filt = np.where((b2m.sum(0) > 0), True, False)
#         i_final = i_nonzero * i_filt

#         diff = np.log10(ec_pred[i_final]) - np.log10(ec_real[i_final]) # / np.log10(ec_real[i_nonzero])
#         pred_error = np.sqrt(np.mean(diff**2)) #np.sqrt(np.dot(pred_error, pred_error.T)) #np.sqrt(np.sum(pred_error**2))

#         rmse_arr.append(pred_error)
#         cf_arr.append(cf)
#         n_produced_arr.append(n_pro)

# # %%
# df = pd.DataFrame({'CT_Freq': cf_arr,
#                    'N_produced': n_produced_arr,
#                    'RMSE': rmse_arr})

# sns.lineplot(data=df, x='N_produced', y='RMSE', hue='CT_Freq', palette='crest')

# # %%
# ###### Randomly chosen subsets of "interconversion"; celltype 1 consumes what celltype 2 produces, and vice versa
# f = 0.55
# in_degree_flag = False

# net, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(all_random_networks[0])
# ec_real = ec_metabolome.iloc[:, 0].values

# rmse_arr = []
# n_produced_arr = []
# cf_arr = []
# rep_arr = []
# ec_pred_arr = [[]]
# index_arr = [[]]
# n_reps = 10
# # count = 1
# for i in np.repeat(np.arange(1, 6), n_reps):
#     max_links = MAX_ID_metabolites * MAX_ID_celltypes
#     n_pro = i*10
#     # count = i
#     ### Edges for celltype 1
#     selected_edges_random = np.random.choice(np.arange(MAX_ID_metabolites), size=n_pro*2, replace=False)
#     consumption_edges = selected_edges_random[:n_pro]
#     production_edges = selected_edges_random[n_pro:]

#     ## Assigning celltype 1 edges
#     net_consumption = net.iloc[:max_links, :]
#     nc1 = net_consumption[net_consumption['celltypes']==0]
#     nc2 = net_consumption[net_consumption['celltypes']==1]
#     nc1.iloc[consumption_edges, -1] = np.repeat(2, n_pro)
#     nc2.iloc[consumption_edges, -1] = np.repeat(0, n_pro)
#     nc1.iloc[~consumption_edges, -1] = np.repeat(0, n_pro)
#     nc2.iloc[~consumption_edges, -1] = np.repeat(2, n_pro)
#     net_consumption = pd.concat([nc1, nc2])

#     net_production = net.iloc[max_links:, :]
#     np1 = net_production[net_production['celltypes']==0]
#     np2 = net_production[net_production['celltypes']==1]
#     np1.iloc[production_edges, -1] = np.repeat(3, n_pro)
#     np2.iloc[production_edges, -1] = np.repeat(0, n_pro)
#     np1.iloc[~production_edges, -1] = np.repeat(0, n_pro)
#     np2.iloc[~production_edges, -1] = np.repeat(3, n_pro)
#     net_production = pd.concat([np1, np2])

#     net_mod = pd.concat([net_consumption, net_production])
#     x = np.where(net_mod.iloc[:, -1] > 0, 1, 0)
#     # edges = np.append(np.zeros(MAX_ID_metabolites-n_pro), np.ones(n_pro))
#     # x[max_links:] = np.random.permutation(np.repeat(edges[np.newaxis, :], 2, axis=0).ravel())
#     net_mod = net_from_x(x, MAX_ID_metabolites=MAX_ID_metabolites, MAX_ID_celltypes=MAX_ID_celltypes)
    
#     for cf in np.linspace(0.1, 0.9, 10):
#         ct_freq = np.array([cf, 1-cf])*cellnum_final_all[0]
#         m2b, b2m, ec_pred = calculate_metabolome_from_net(f, ct_freq, diet, net_mod, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

#         i_nonzero = np.where(ec_pred * ec_real, True, False)
#         i_filt = np.where((b2m.sum(0) > 0), True, False)
#         i_final = i_nonzero * i_filt

#         diff = np.log10(ec_pred[i_final]) - np.log10(ec_real[i_final]) # / np.log10(ec_real[i_nonzero])
#         pred_error = np.sqrt(np.mean(diff**2)) #np.sqrt(np.dot(pred_error, pred_error.T)) #np.sqrt(np.sum(pred_error**2))

#         rmse_arr.append(pred_error)
#         cf_arr.append(cf)
#         n_produced_arr.append(n_pro)
#         ec_pred_arr.append(ec_pred)
#         index_arr.append(i_final)
#     # rep_arr.append(np.repeat(count, n_reps))
#     # count += 1s

# ec_pred_arr = np.array(ec_pred_arr[1:])
# index_arr = np.array(index_arr[1:])
# # %%
# df = pd.DataFrame({'CT_Freq': cf_arr,
#                    'N_produced': n_produced_arr,
#                 #    'Replicate': np.array(rep_arr),
#                    'RMSE': rmse_arr})

# sns.pointplot(data=df, x='CT_Freq', y='RMSE', hue='N_produced', palette='crest',
#              markers=True, estimator='mean', errorbar='sd')

# # %%
# ###### Partition metabolome into high and low, celltypes are random consumers of the union of metabolites, but partition production into high vs low mets
# f = 1
# in_degree_flag = False
# home_dir = os.getcwd()
# all_networks, i_intake, names = pd.read_pickle(home_dir + '/1-cells-cancer_network.pickle')

# # net, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(all_networks[0])

# rmse_arr = []
# n_produced_arr = []
# cf_arr = []
# rep_arr = []
# ec_pred_arr = [[]]
# index_arr = [[]]
# production_ct_arr = [[]]
# consumption_ct_arr = [[]]
# nct_arr = []
# # max_links = MAX_ID_metabolites * MAX_ID_celltypes

# n_reps = 100
# npro_arr = np.repeat(np.array([50])[np.newaxis, :], n_reps*2, axis=1).ravel()
# nct_arr = np.repeat(np.array([[1]*n_reps, [2]*n_reps]).ravel()[np.newaxis, :], 1, axis=0).ravel()

# for n_pro, n_ct in zip(npro_arr, nct_arr):#np.repeat(np.arange(1, 6), n_reps*2):
#     # n_pro = i*10
#     home_dir = os.getcwd()
#     all_networks, i_intake, names = pd.read_pickle(home_dir + '/' + str(n_ct) + '-cells-cancer_network.pickle')
#     # i_selfish = 0

#     # pickle_in = open("data.pickle","rb")
#     celltype_ID, celltypefreq, ec_metabolome_ID, ec_metabolome, met_baseline, core_mean = pd.read_pickle(home_dir + '/' + str(n_ct) + '-cells-data.pickle')
#     cell_line_names = ec_metabolome.columns.to_numpy()

#     ## Source for initial and final cell numbers for the various cell lines: https://www.thermofisher.com/in/en/home/references/gibco-cell-culture-basics/cell-culture-protocols/cell-culture-useful-numbers.html
#     n_lines = len(core_mean.loc[:, 'Cell line'])
#     # nonadh_cell_lines = ['SR', 'MOLT-4', 'HL-60(TB)', 'K562', 'RPMI 8226', 'CCRF-CEM']
#     # i_nonadh = np.where(np.isin(core_mean.loc[:, 'Cell line'], nonadh_cell_lines))[0]

#     t75_cell_lines = ['NCI-H460', 'HCC-2998', 'SW620']
#     i_t75 = np.where(np.isin(cell_line_names, t75_cell_lines))[0]
#     cellnum_init_all = np.array([4.9e+06]*n_lines)
#     cellnum_init_all[i_t75] = 2.1e+06
#     cellnum_final_all = np.array([23.3e+06]*n_lines)
#     cellnum_final_all[i_t75] = 8.4e+06

#     slope_df, sublinear_cell_lines = pd.read_pickle(home_dir + '/cancer_power_law_stats.pickle')
#     nets_temp = all_networks.copy()

#     ### Filtering those networks for cell lines with power law slopes
#     i_sublinear = np.where(np.isin(ec_metabolome.columns.values, sublinear_cell_lines))[0]
#     all_networks = []
#     for i in i_sublinear:
#         all_networks.append(nets_temp[i])
#     net, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(all_networks[0])

#     ec_metabolome = ec_metabolome.iloc[:, i_sublinear]
#     diet = met_baseline.mean(axis=1)
#     ec_real = ec_metabolome.iloc[:, 0].values
#     low_mets = np.where(ec_real <= np.median(ec_real))[0]
#     high_mets = np.where(ec_real > np.median(ec_real))[0]

#     i_nonzero_celltypes = all_networks[0]['celltypes_ID'].unique()
#     i_nonzero_celltypes = np.sort(i_nonzero_celltypes)
#     i_nonzero_celltypes = celltype_ID.values.copy()
#     i_nonzero_metabolites = all_networks[0]['metabolites_ID'].unique()
#     # i_nonzero_metabolites = np.sort(i_nonzero_metabolites)

#     cellnum_init_all = cellnum_init_all[i_sublinear]
#     cellnum_final_all = cellnum_final_all[i_sublinear]

#     MAX_ID_celltypes = len(i_nonzero_celltypes)  # MAX_ID_celltypes is the maximum of ID labels for celltypes.
#     MAX_ID_metabolites = len(i_nonzero_metabolites)  # MAX_ID_metabolites is the maximum of ID labels for metabolites.
#     max_links = MAX_ID_celltypes*MAX_ID_metabolites

#     ### I'm generating the randomized edge lists assuming a default two-celltype case because it's easier to code.
#     ### Partition edges for production
#     rand = np.random.randint(low=1, high=n_pro)
#     num_prod_ct1 = rand
#     num_prod_ct2 = n_pro - rand
#     pro_edges_ct1 = np.random.choice(low_mets, size=num_prod_ct1, replace=False)
#     pro_edges_ct2 = np.random.choice(high_mets, size=num_prod_ct2, replace=False)

#     ### Within the interconversion regime, consumption is random
#     rand = np.random.randint(low=1, high=n_pro)
#     num_con_ct1 = rand
#     num_con_ct2 = n_pro-rand
#     all_edges = np.random.permutation(np.concatenate([pro_edges_ct1, pro_edges_ct2]))
#     con_edges_ct1 = all_edges[:rand]
#     con_edges_ct2 = all_edges[rand:]
#     # all_edges = np.arange(MAX_ID_metabolites)
#     # con_edges_ct1 = np.random.choice(all_edges, size=num_con_ct1, replace=False)
#     # con_edges_ct2 = np.random.choice(all_edges, size=num_con_ct2, replace=False)

#     ### Assigning edges-seperated by celltype for two celltypes and the union of whatever is selected for the single celltype
#     if n_ct == 2:
#         net_consumption = net.iloc[:max_links, :]
#         net_consumption.iloc[:, -1] = np.zeros(max_links)
#         nc1 = net_consumption[net_consumption['celltypes']==0]
#         nc2 = net_consumption[net_consumption['celltypes']==1]
#         nc1.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), con_edges_ct1), 2, 0)
#         nc2.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), con_edges_ct2), 2, 0)   
#         net_consumption = pd.concat([nc1, nc2])

#         net_production = net.iloc[max_links:, :]
#         net_production.iloc[:, -1] = np.zeros(max_links)
#         np1 = net_production[net_production['celltypes']==0]
#         np2 = net_production[net_production['celltypes']==1]
#         np1.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), pro_edges_ct1), 3, 0)
#         np2.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), pro_edges_ct2), 3, 0)
#         net_production = pd.concat([np1, np2])

#         net_mod = pd.concat([net_consumption, net_production])

#         cf = np.round(np.random.uniform(0, 1), decimals=2)
#         ct0 = np.array([cf, 1-cf])*cellnum_init_all[0]

#     if n_ct == 1:
#         con_edges_union = np.unique(np.concatenate([con_edges_ct1, con_edges_ct2]))#np.random.choice(np.arange(MAX_ID_metabolites), n_pro, replace=False)
#         net_consumption = net.iloc[:max_links, :]
#         net_consumption.iloc[:, -1] = np.zeros(max_links)
#         net_consumption.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), con_edges_union), 2, 0)

#         pro_edges_union = con_edges = np.unique(np.concatenate([pro_edges_ct1, pro_edges_ct2]))#np.random.choice(np.arange(MAX_ID_metabolites), n_pro, replace=False)
#         net_production = net.iloc[max_links:, :]
#         net_production.iloc[:, -1] = np.zeros(max_links)
#         net_production.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), pro_edges_union), 3, 0)
        
#         net_mod = pd.concat([net_consumption, net_production])
#         ct0 = np.array([cellnum_init_all[0]])

#     ### Learn the best celltype frequency
#     ct_final = np.zeros_like(ct0) # Final cell number, either fitted or taken depending on number of cell types
#     ##### For max celltypes > 1, the model is converted into an optimization problem where the celltype frequencies are learned to minimize the logarithmic error between experimentally measured metabolome and predicted metabolome computed from the model for a certain up-sec network and initial cell type distribution.
#     if MAX_ID_celltypes > 1:
#         my_args = (net_mod, f, diet, ec_real, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
#         bnds = ((0, cellnum_final_all[0]), ) * len(ct0)
#         constraint = {'type': 'eq', 'fun': lambda ct: ct.sum() - cellnum_final_all[0]}
#         res = minimize(calc_pred_error, ct0, args=my_args, method='SLSQP', bounds=bnds, options={'disp': False, 'maxiter': 1000}, tol=1e-3, constraints=constraint)
#         ct_final = res.x #res.x.max()/cellnum_max
        
#     #### For max celltypes = 1, no model fitting, final cell number is taken directly from cell number at confluency
#     else:
#         ct_final = cellnum_final_all[0]*celltypefreq.values

#     ### Calculate predicted metabolome using the fitted celltype frequencies
#     m2b, b2m, ec_pred = calculate_metabolome_from_net(f, ct_final, diet, net_mod, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
#     cf_final = ct_final[0]/ct_final.sum()

#     i_nonzero = np.where(ec_pred * ec_real, True, False)
#     i_filt = np.where((b2m.sum(0) > 0), True, False)
#     i_final = i_nonzero * i_filt

#     diff = np.log10(ec_pred[i_final]) - np.log10(ec_real[i_final]) # / np.log10(ec_real[i_nonzero])
#     pred_error = np.sqrt(np.mean(diff**2)) #np.sqrt(np.dot(pred_error, pred_error.T)) #np.sqrt(np.sum(pred_error**2))
    
#     p_arr = np.array([i*j for i, j in zip(b2m, [1, 2])]).sum(0)
#     c_arr = np.array([i*j for i, j in zip(m2b, [1, 2])]).sum(0)
#     rmse_arr.append(pred_error)
#     cf_arr.append(cf_final.round(decimals=2))
#     n_produced_arr.append(n_pro)
#     ec_pred_arr.append(ec_pred)
#     index_arr.append(i_final)
#     production_ct_arr.append(p_arr)
#     consumption_ct_arr.append(c_arr)

# ec_pred_arr = np.array(ec_pred_arr[1:])
# index_arr = np.array(index_arr[1:])
# production_ct_arr = np.array(production_ct_arr[1:])
# consumption_ct_arr = np.array(consumption_ct_arr[1:])
# rmse_arr = np.array(rmse_arr)
# cf_arr = np.array(cf_arr)
# n_produced_arr = np.array(n_produced_arr)

# # %%
# f, ax = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(6, 9))
# index_1ct = np.where(nct_arr == 1, True, False)
# index_2ct = np.where(nct_arr == 2, True, False)

# #### First column-single celltype
# pct_arr = production_ct_arr[index_1ct]
# pred_arr = ec_pred_arr[index_1ct]
# index = index_arr[index_1ct]
# npro = n_produced_arr[index_1ct]
# rmse = rmse_arr[index_1ct]

# i_10mets = np.where(npro==10, True, False)
# i_best_10 = np.where(rmse == np.min(rmse[i_10mets]))[0][0]
# colors = np.where(pct_arr[i_best_10], 'tab:blue', 'tab:green')
# labels = np.where(pct_arr[i_best_10], 'CT1', 'CT2')
# ax[0, 0].scatter(np.log10(pred_arr[i_best_10][index[i_best_10]]), np.log10(ec_real[index[i_best_10]]),
#               c=colors[index[i_best_10]],#'tab:blue',
#               alpha=0.6, edgecolor='face')
# ax[0, 0].axline((-2, -2), (3, 3), c='k', ls='--')
# ax[0, 0].set_title(r'$N_{produced} = $'+str(npro[i_best_10]))
# ax[0, 0].text(0.05, 0.9, f'RMSE = {rmse[i_best_10]:.2f}', transform=ax[0, 0].transAxes)

# i_20mets = np.where(npro==20, True, False)
# i_best_20 = np.where(rmse == np.min(rmse[i_20mets]))[0][0]
# colors = np.where(pct_arr[i_best_20], 'tab:blue', 'tab:green')
# labels = np.where(pct_arr[i_best_20], 'CT1', 'CT2')
# ax[1, 0].scatter(np.log10(pred_arr[i_best_20][index[i_best_20]]), np.log10(ec_real[index[i_best_20]]),
#               c=colors[index[i_best_20]],#'tab:blue',
#               alpha=0.6, edgecolor='face')
# ax[1, 0].axline((-2, -2), (3, 3), c='k', ls='--')
# ax[1, 0].set_title(r'$N_{produced} = $'+str(npro[i_best_20]))
# ax[1, 0].text(0.05, 0.9, f'RMSE = {rmse[i_best_20]:.2f}', transform=ax[1, 0].transAxes)

# i_50mets = np.where(npro==50, True, False)
# i_best_50 = np.where(rmse == np.min(rmse[i_50mets]))[0][0]
# colors = np.where(pct_arr[i_best_50], 'tab:blue', 'tab:green')
# labels = np.where(pct_arr[i_best_50], 'CT1', 'CT2')
# ax[2, 0].scatter(np.log10(pred_arr[i_best_50][index[i_best_50]]), np.log10(ec_real[index[i_best_50]]),
#               c=colors[index[i_best_50]],#'tab:blue',
#               alpha=0.6, edgecolor='face')
# ax[2, 0].axline((-2, -2), (3, 3), c='k', ls='--')
# ax[2, 0].set_title(r'$N_{produced} = $'+str(npro[i_best_50]))
# ax[2, 0].text(0.05, 0.9, f'RMSE = {rmse[i_best_50]:.2f}', transform=ax[2, 0].transAxes)

# #### Second column-two celltype
# pct_arr = production_ct_arr[index_2ct]
# pred_arr = ec_pred_arr[index_2ct]
# index = index_arr[index_2ct]
# npro = n_produced_arr[index_2ct]
# rmse = rmse_arr[index_2ct]
# cf = cf_arr[index_2ct]

# i_10mets = np.where(npro==10, True, False)
# i_best_10 = np.where(rmse == np.min(rmse[i_10mets]))[0][0]
# colors = np.where(pct_arr[i_best_10], 'tab:blue', 'tab:green')
# labels = np.where(pct_arr[i_best_10], 'CT1', 'CT2')
# ax[0, 1].scatter(np.log10(pred_arr[i_best_10][index[i_best_10]]), np.log10(ec_real[index[i_best_10]]),
#               c=colors[index[i_best_10]],#'tab:blue',
#               alpha=0.6, edgecolor='face')
# ax[0, 1].axline((-2, -2), (3, 3), c='k', ls='--')
# ax[0, 1].set_title(r'$N_{produced} = $'+str(npro[i_best_10]))
# ax[0, 1].text(0.05, 0.9, f'RMSE = {rmse[i_best_10]:.2f}', transform=ax[0, 1].transAxes)
# ax[0, 1].text(0.05, 0.8, r'$\chi_{CT1}$ = '+f'{cf[i_best_10]:.2f}', transform=ax[0, 1].transAxes)

# i_20mets = np.where(npro==20, True, False)
# i_best_20 = np.where(rmse == np.min(rmse[i_20mets]))[0][0]
# colors = np.where(pct_arr[i_best_20], 'tab:blue', 'tab:green')
# labels = np.where(pct_arr[i_best_20], 'CT1', 'CT2')
# ax[1, 1].scatter(np.log10(pred_arr[i_best_20][index[i_best_20]]), np.log10(ec_real[index[i_best_20]]),
#               c=colors[index[i_best_20]],#'tab:blue',
#               alpha=0.6, edgecolor='face')
# ax[1, 1].axline((-2, -2), (3, 3), c='k', ls='--')
# ax[1, 1].set_title(r'$N_{produced} = $'+str(npro[i_best_20]))
# ax[1, 1].text(0.05, 0.9, f'RMSE = {rmse[i_best_20]:.2f}', transform=ax[1, 1].transAxes)
# ax[1, 1].text(0.05, 0.8, r'$\chi_{CT1}$ = '+f'{cf[i_best_20]:.2f}', transform=ax[1, 1].transAxes)

# i_50mets = np.where(npro==50, True, False)
# i_best_50 = np.where(rmse == np.min(rmse[i_50mets]))[0][0]
# colors = np.where(pct_arr[i_best_50], 'tab:blue', 'tab:green')
# labels = np.where(pct_arr[i_best_50], 'CT1', 'CT2')
# ax[2, 1].scatter(np.log10(pred_arr[i_best_50][index[i_best_50]]), np.log10(ec_real[index[i_best_50]]),
#               c=colors[index[i_best_50]],#'tab:blue',
#               alpha=0.6, edgecolor='face')
# ax[2, 1].axline((-2, -2), (3, 3), c='k', ls='--')
# ax[2, 1].set_title(r'$N_{produced} = $'+str(npro[i_best_50]))
# ax[2, 1].text(0.05, 0.9, f'RMSE = {rmse[i_best_50]:.2f}', transform=ax[2, 1].transAxes)
# ax[2, 1].text(0.05, 0.8, r'$\chi_{CT1}$ = '+f'{cf[i_best_50]:.2f}', transform=ax[2, 1].transAxes)

# f.supxlabel(r'$log_{10}\ Predicted\ metabolome$')
# f.supylabel(r'$log_{10}\ Empirical\ metabolome$')
# f.tight_layout()

# # %%
# df = pd.DataFrame({'Replicate': np.repeat(np.arange(1, n_reps+1)[np.newaxis, :], 2, axis=0).ravel(),
#                     'N_CT': nct_arr,
#                    'N_produced': n_produced_arr,
#                    'CT_Freq': cf_arr,
#                    'RMSE': rmse_arr})
# # df['CT_Freq'] = df['CT_Freq'].astype('category')

# g = sns.pointplot(data=df, x='N_produced', y='RMSE', hue='N_CT', palette='crest',
#                 estimator='mean', errorbar='sd', capsize=0.1)
# g.set_xlabel(r'$N_{produced}$')

# h = g.twinx()
# h = sns.pointplot(data=df[df['N_CT']==2], x='N_produced', y='CT_Freq',
#                   estimator='mean', errorbar='sd', capsize=0.1,
#                   color='tab:red', linestyles='--', marker='D',
#                   alpha=0.8)
# h.set_ylabel(r'$\chi_{CT1}$', color='tab:red')
# h.axes.tick_params(axis='y', colors='tab:red')
# h.axes.spines['right'].set_color('tab:red')
# plt.show()

# for npro in [50]:
#     # colors = plt.cm.gist_yarg(np.linspace(0, 0.7, n_reps))
#     g = sns.lineplot(data=df[df['N_produced']==npro], x='N_CT', y='RMSE',
#                 units='Replicate',
#                 dashes=False, estimator=None)
#     g = sns.scatterplot(data=df[df['N_produced']==npro], x='N_CT', y='RMSE',
#                         edgecolor='face', alpha=0.8)
#     g.set_xlabel(r'$N_{CT}$')
#     g.set(xlim=(0.5, 2.5), xticks=[1, 2])
#     g.set_title(r'$N_{produced}=$'+str(npro))
#     plt.show()

# # %%
# df.loc[:, 'Npro_CT1'] = np.where(production_ct_arr == 1, ec_pred_arr, 0).sum(1)#production_ct_arr.sum(1)
# df.loc[:, 'Npro_CT2'] = np.where(production_ct_arr == 2, ec_pred_arr, 0).sum(1)
# # weighted_production = np.where(production_ct_arr, ec_real, 0).sum(1)
# # df.loc[:, 'Xpro_CT1'] = weighted_production/ec_real.sum()

# df.loc[:, 'Ncon_CT1'] = np.where(consumption_ct_arr == 1, diet.values, 0).sum(1)#production_ct_arr.sum(1)
# df.loc[:, 'Ncon_CT2'] = np.where(consumption_ct_arr == 2, diet.values, 0).sum(1)

# df.loc[:, 'Xpro_CT1'] = df.loc[:, 'Npro_CT1'].values/df.loc[:, 'Ncon_CT1'].values
# df.loc[:, 'Xpro_CT2'] = df.loc[:, 'Npro_CT2'].values/df.loc[:, 'Ncon_CT2'].values

# # %%
# g = sns.scatterplot(data=df[df['N_CT']==2], x='Xpro_CT1', y='RMSE', markers=['o', 's', '^', 'X', 'P'],
#             hue='N_produced', palette='crest',
#             edgecolor= 'face', alpha = 0.9, s = 30)
# g.set_xlabel(r'$\chi_{production, CT1}$')
# # g.set(xscale='log')
# g.tick_params(axis='x', labelrotation=45)
# plt.show()

# g = sns.scatterplot(data=df[df['N_CT']==2], x='Xpro_CT2', y='RMSE', markers=['o', 's', '^', 'X', 'P'],
#             hue='N_produced', palette='crest',
#             edgecolor= 'face', alpha = 0.9, s = 30)
# g.set_xlabel(r'$\chi_{production, CT2}$')
# # g.set(xscale='log')
# g.tick_params(axis='x', labelrotation=45)
# plt.show()

# g = sns.scatterplot(data=df[df['N_CT']==2], x='Xpro_CT1', y='Xpro_CT2',
#                     hue = 'RMSE', palette='coolwarm', #sizes=(50, 200),
#             # hue='N_produced', palette='crest',
#             edgecolor= 'face', alpha = 0.9, s=75)
# g.axhline(y=1, linestyle='dashed', c='tab:green')
# g.text(0.9, 0.38, r'$y=1$', c='tab:green', transform=g.transAxes)
# g.axvline(x=1, linestyle='dashed', c='tab:green')
# g.text(0.68, 0.87, r'$x=1$', c='tab:green', rotation='vertical', transform=g.transAxes)

# i_min_error = np.where(df['RMSE'] == df[df['N_CT']==2]['RMSE'].min())[0][0]
# g.axhline(y=df['Xpro_CT2'][i_min_error], linestyle='dashed', c='tab:blue')
# g.text(0.79, 0.3, r'RMSE min', c='tab:blue', transform=g.transAxes)
# g.axvline(x=df['Xpro_CT1'][i_min_error], linestyle='dashed', c='tab:blue')
# g.text(0.62, 0.75, r'RMSE min', c='tab:blue', rotation='vertical', transform=g.transAxes)

# # g.axhline(y=)
# g.set_xlabel(r'$\chi_{production, CT1}$')
# g.set_ylabel(r'$\chi_{production, CT2}$')
# g.set(xscale='log', yscale='log')
# g.tick_params(axis='x', labelrotation=45)
# plt.show()

# # %%
# df.loc[:, 'Ncon_CT1'] = consumption_ct_arr.sum(1)
# weighted_consumption = np.where(consumption_ct_arr, diet.values, 0).sum(1)
# df.loc[:, 'Xcon_CT1'] = weighted_consumption/diet.values.sum()

# g = sns.lmplot(data=df[df['N_CT']==2], x='Ncon_CT1', y='RMSE', markers=['o', 's', '^', 'X', 'P'],
#             hue='N_produced', palette='crest',
#             scatter_kws={'edgecolor': 'face', 'alpha': 0.75, 's': 35})
# g.ax.set_xlabel(r'$N_{consumed, CT1}$')
# # g.ax.set(xscale='log')
# g.tick_params(axis='x', labelrotation=45)

# # %%
# for npro in [10, 30, 50]:
#     f, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(5.5, 5.5))

#     # npro = 10
#     mets_index = np.where(n_produced_arr == npro, True, False)
#     pct_arr = production_ct_arr[mets_index]
#     con_arr = consumption_ct_arr[mets_index]
#     pred_arr = ec_pred_arr[mets_index]
#     index = index_arr[mets_index]
#     nct = nct_arr[mets_index]
#     rmse = rmse_arr[mets_index]
#     cf = cf_arr[mets_index]

#     i_1ct = np.where(nct==1, True, False)
#     i_2ct = np.where(nct==2, True, False)

#     rmse_diff = rmse[i_2ct] - rmse[i_1ct]
#     i_min_diff = np.where(rmse_diff == rmse_diff.min(), True, False)
#     i_max_diff = np.where(rmse_diff == rmse_diff.max(), True, False)

#     #### Minimum difference
#     colors = np.where(pct_arr[i_1ct][i_min_diff]==1, 'tab:blue', 'tab:green')
#     labels = np.where(pct_arr[i_1ct][i_min_diff]==1, 'CT1', 'CT2')
#     ax[0, 0].scatter(np.log10(pred_arr[i_1ct][i_min_diff][index[i_1ct][i_min_diff]]), 
#                     np.log10([ec_real])[index[i_1ct][i_min_diff]],
#                 c=colors[index[i_1ct][i_min_diff]],#'tab:blue',
#                 alpha=0.6, edgecolor='face')
#     ax[0, 0].axline((-2, -2), (3, 3), c='k', ls='--')
#     ax[0, 0].set_title(r'$N_{CT} = 1$')
#     ax[0, 0].text(0.05, 0.9, f'RMSE = {rmse[i_1ct][i_min_diff][0]:.2f}', transform=ax[0, 0].transAxes)

#     colors = np.where(pct_arr[i_2ct][i_min_diff]==1, 'tab:blue', 'tab:green')
#     labels = np.where(pct_arr[i_2ct][i_min_diff]==1, 'CT1', 'CT2')
#     con_index_ct1 = np.where(con_arr[i_2ct][i_min_diff]==1, True, False)[0]*index[i_2ct][i_min_diff]
#     con_index_ct2 = np.where(con_arr[i_2ct][i_min_diff]==2, True, False)[0]*index[i_2ct][i_min_diff]
#     ax[0, 1].scatter(np.log10(pred_arr[i_2ct][i_min_diff][con_index_ct1]), 
#                     np.log10([ec_real])[con_index_ct1],
#                 c=colors[con_index_ct1], marker='o',
#                 alpha=0.6, edgecolor='face')
#     ax[0, 1].scatter(np.log10(pred_arr[i_2ct][i_min_diff][con_index_ct2]), 
#                     np.log10([ec_real])[con_index_ct2],
#                 c=colors[con_index_ct2], marker='X',
#                 alpha=0.6, edgecolor='face')
#     ax[0, 1].axline((-2, -2), (3, 3), c='k', ls='--')
#     ax[0, 1].set_title(r'$N_{CT} = 2$')
#     ax[0, 1].text(0.05, 0.9, f'RMSE = {rmse[i_2ct][i_min_diff][0]:.2f}', transform=ax[0, 1].transAxes)
#     ax[0, 1].text(0.05, 0.8, r'$\chi_{CT1}$ = '+f'{cf[i_2ct][i_min_diff][0]:.2f}', transform=ax[0, 1].transAxes)

#     #### Maximum difference
#     colors = np.where(pct_arr[i_1ct][i_max_diff]==1, 'tab:blue', 'tab:green')
#     labels = np.where(pct_arr[i_1ct][i_max_diff]==1, 'CT1', 'CT2')
#     ax[1, 0].scatter(np.log10(pred_arr[i_1ct][i_max_diff][index[i_1ct][i_max_diff]]), 
#                     np.log10([ec_real])[index[i_1ct][i_max_diff]],
#                 c=colors[index[i_1ct][i_max_diff]],#'tab:blue',
#                 alpha=0.6, edgecolor='face')
#     ax[1, 0].axline((-2, -2), (3, 3), c='k', ls='--')
#     ax[1, 0].text(0.05, 0.9, f'RMSE = {rmse[i_1ct][i_max_diff][0]:.2f}', transform=ax[1, 0].transAxes)

#     colors = np.where(pct_arr[i_2ct][i_max_diff]==1, 'tab:blue', 'tab:green')
#     labels = np.where(pct_arr[i_2ct][i_max_diff]==1, 'CT1', 'CT2')
#     con_index_ct1 = np.where(con_arr[i_2ct][i_max_diff]==1, True, False)[0]*index[i_2ct][i_max_diff]
#     con_index_ct2 = np.where(con_arr[i_2ct][i_max_diff]==2, True, False)[0]*index[i_2ct][i_max_diff]
#     ax[1, 1].scatter(np.log10(pred_arr[i_2ct][i_max_diff][con_index_ct1]), 
#                     np.log10([ec_real])[con_index_ct1],
#                 c=colors[con_index_ct1], marker='o',
#                 alpha=0.6, edgecolor='face')
#     ax[1, 1].scatter(np.log10(pred_arr[i_2ct][i_max_diff][con_index_ct2]), 
#                     np.log10([ec_real])[con_index_ct2],
#                 c=colors[con_index_ct2], marker='X',
#                 alpha=0.6, edgecolor='face')
#     ax[1, 1].axline((-2, -2), (3, 3), c='k', ls='--')
#     ax[1, 1].text(0.05, 0.9, f'RMSE = {rmse[i_2ct][i_max_diff][0]:.2f}', transform=ax[1, 1].transAxes)
#     ax[1, 1].text(0.05, 0.8, r'$\chi_{CT1}$ = '+f'{cf[i_2ct][i_max_diff][0]:.2f}', transform=ax[1, 1].transAxes)
    
#     f.supxlabel(r'$log_{10}\ Predicted\ metabolome$')
#     f.supylabel(r'$log_{10}\ Empirical\ metabolome$')
#     f.suptitle(r'$N_{produced} = $'+str(npro))
#     f.tight_layout()

#################################################################
### cancer-no-learning-networks
    # pro_edges_ct1 = chosen_edges[np.isin(chosen_edges, low_mets)]
    # pro_edges_ct2 = chosen_edges[np.isin(chosen_edges, high_mets)]

    # # ### Production edges without partition
    # # rand = np.random.randint(low=1, high=npro)
    # # num_prod_ct1 = rand
    # # num_prod_ct2 = npro - rand
    # # pro_edges_ct1 = np.random.choice(chosen_edges, size=num_prod_ct1, replace=False)
    # # pro_edges_ct2 = np.random.choice(chosen_edges, size=num_prod_ct2, replace=False)

    # rand = np.random.randint(low=1, high=npro)
    # num_con_ct1 = rand
    # num_con_ct2 = npro-rand
    # all_edges = np.random.permutation(np.concatenate([pro_edges_ct1, pro_edges_ct2]))
    # con_edges_ct1 = all_edges[:rand]
    # con_edges_ct2 = all_edges[rand:]

    # ### Assigning edges to the two celltype network
    # nc1 = net_consumption[net_consumption['celltypes']==0]
    # nc2 = net_consumption[net_consumption['celltypes']==1]
    # nc1.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), con_edges_ct1), 2, 0)
    # nc2.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), con_edges_ct2), 2, 0)   
    # net_consumption = pd.concat([nc1, nc2])

    # np1 = net_production[net_production['celltypes']==0]
    # np2 = net_production[net_production['celltypes']==1]
    # np1.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), pro_edges_ct1), 3, 0)
    # np2.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), pro_edges_ct2), 3, 0)
    # net_production = pd.concat([np1, np2])

        # # ### Production edges without partition
        # # rand = np.random.randint(low=1, high=npro)
        # # num_prod_ct1 = rand
        # # num_prod_ct2 = npro - rand
        # # pro_edges_ct1 = np.random.choice(chosen_edges, size=num_prod_ct1, replace=False)
        # # pro_edges_ct2 = np.random.choice(chosen_edges, size=num_prod_ct2, replace=False)

        # ### Within the interconversion regime, consumption is random
        # rand = np.random.randint(low=1, high=npro)
        # num_con_ct1 = rand
        # num_con_ct2 = npro-rand
        # all_edges = np.random.permutation(np.concatenate([pro_edges_ct1, pro_edges_ct2]))
        # con_edges_ct1 = all_edges[:rand]
        # con_edges_ct2 = all_edges[rand:]

        # ### Assigning edges to the two celltype network
        # nc1 = net_consumption[net_consumption['celltypes']==0]
        # nc2 = net_consumption[net_consumption['celltypes']==1]
        # nc1.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), con_edges_ct1), 2, 0)
        # nc2.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), con_edges_ct2), 2, 0)   
        # net_consumption = pd.concat([nc1, nc2])

        # np1 = net_production[net_production['celltypes']==0]
        # np2 = net_production[net_production['celltypes']==1]
        # np1.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), pro_edges_ct1), 3, 0)
        # np2.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), pro_edges_ct2), 3, 0)
        # net_production = pd.concat([np1, np2])

    # ### Assigning edges to the two celltype network
    # nc1 = net_consumption[net_consumption['celltypes']==0]
    # nc2 = net_consumption[net_consumption['celltypes']==1]
    # nc1.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), con_edges_ct1), 2, 0)
    # nc2.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), con_edges_ct2), 2, 0)   
    # net_consumption = pd.concat([nc1, nc2])

    # np1 = net_production[net_production['celltypes']==0]
    # np2 = net_production[net_production['celltypes']==1]
    # np1.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), pro_edges_ct1), 3, 0)
    # np2.iloc[:, -1] = np.where(np.isin(np.arange(MAX_ID_metabolites), pro_edges_ct2), 3, 0)
    # net_production = pd.concat([np1, np2])

    # net_2ct = pd.concat([net_consumption, net_production])

    # cf = np.round(np.random.uniform(0, 1), decimals=2)
    # ct0_2ct = np.array([cf, 1-cf])*cellnum_init_all[0]

# con_index_ct1 = np.where(consumption_ct_arr_2ct[i_min_diff]==0, True, False)[0]*index_arr_2ct[i_min_diff]
# con_index_ct2 = np.where(consumption_ct_arr_2ct[i_min_diff]==1, True, False)[0]*index_arr_2ct[i_min_diff]
# ax[1].scatter(np.log10(ec_pred_arr_2ct[i_min_diff][con_index_ct2]), 
#                 np.log10(ec_real[con_index_ct2[0]]),
#             c=colors[con_index_ct2], marker='X',
#             alpha=0.6, edgecolor='face')

# con_index_ct1 = np.where(consumption_ct_arr_2ct[i_max_diff]==0, True, False)[0]*index_arr_2ct[i_max_diff]
# con_index_ct2 = np.where(consumption_ct_arr_2ct[i_max_diff]==1, True, False)[0]*index_arr_2ct[i_max_diff]

# ax[2].scatter(np.log10(ec_pred_arr_2ct[i_max_diff][con_index_ct2]), 
#                 np.log10(ec_real[con_index_ct2[0]]),
#             c=colors[con_index_ct2], marker='X',
#             alpha=0.6, edgecolor='face')
