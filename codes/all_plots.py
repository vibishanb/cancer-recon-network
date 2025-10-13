def plot_all_corrs(net_state, fig_name, k, f, metabolome_pred, metabolome_measured, ec_metabolome, figsave_flag=False, disp_flag=True):

    try:
        os.makedirs('../figures/'+str(k)+'-celltypes/'+net_state)
    except:
        pass

    nrow = 10
    ncol = 6
    fig, ax = plt.subplots(nrow, ncol, figsize=(11, 11))
    cl_names = ec_metabolome.columns.to_numpy()
    count = 0
    for i in range(nrow):
        for j in range(ncol):
            if count < len(cl_names):
                ax[i, j].scatter(np.log10(metabolome_pred[count]+1e-7), np.log10(metabolome_measured[count]+1e-7), c='k', s=4)
                ax[i, j].plot([-7, 1], [-7, 1],'k-')
                ax[i, j].set_title('%s' % cl_names[count], {'fontsize': 6}, pad=2)
                ax[i, j].tick_params(axis='both', which='major', length=4, labelsize=10)
                if i < nrow-1:
                    ax[i, j].tick_params(axis='x', which='both', labelbottom=False)
                if j > 0:
                    ax[i, j].tick_params(axis='y', which='both', labelleft=False)
                count += 1
                continue
            else:
                break
    #### Turn off axis labels for the extra three panels in the last row
    for a in ax[-1, -3:]:
        a.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
    fig.supxlabel(r'$log_{10}\ Predicted\ metabolome$', y=0.05)
    fig.supylabel(r'$log_{10}\ Empirical\ data$', x=0.05)
    fig.suptitle(fig_name, y=0.92)
    file_name = str(k)+'-celltypes/'+net_state+fig_name+'-f-'+str(f)+'.png'
    if figsave_flag:
        fig.savefig('../figures/'+file_name, dpi=300)
    if not disp_flag:
        plt.close(fig)

def plot_summary_stats(net_state, fig_name, k, f_arr, ec_corr, ct_full, mean_error, slopes, intercepts, figsave_flag=False, disp_flag=True):
    ##### Correlation coefficient
    figcorr, ax = plt.subplots(figsize=(10, 10))
    corr_df = pd.DataFrame(ec_corr.T, columns=f_arr, index=cell_line_names)
    sns.heatmap(data=corr_df, linewidth=0.5, cmap='crest', ax=ax, vmin=0, vmax=1)
    file_name = str(k)+'-celltypes/'+net_state+fig_name+'-ec-met-correlations.png'
    plt.title('Pearsons r: Extracellular conc')
    plt.xlabel('Byproduct fraction, f')
    plt.ylabel('Cell line')
    if figsave_flag:
        figcorr.savefig('../figures/'+file_name, dpi=300)
    if not disp_flag:
        plt.close(figcorr)

    ##### Predicted max cell type frequency
    # # For single cell type
    # ct_full = ct_full.reshape(ec_corr.shape)
    figct, ax = plt.subplots(figsize=(10, 10))
    ct_df = pd.DataFrame(ct_full.T, columns=f_arr, index=cell_line_names)
    sns.heatmap(data=ct_df, linewidth=0.5, cmap='crest', ax=ax, vmin=0, vmax=1)
    file_name = str(k)+'-celltypes/'+net_state+fig_name+'-celltype-freq.png'
    plt.title('Cell type abundance')
    plt.xlabel('Byproduct fraction, f')
    plt.ylabel('Cell line')
    if figsave_flag:
        figct.savefig('../figures/'+file_name, dpi=300)
    if not disp_flag:
        plt.close(figct)

    ##### Mean error of metabolome prediction
    figerror, ax = plt.subplots(figsize=(10, 10))
    err_df = pd.DataFrame(mean_error.T, columns=f_arr, index=cell_line_names)
    sns.heatmap(data=err_df, linewidth=0.5, cmap='crest', ax=ax)
    file_name = str(k)+'-celltypes/'+net_state+fig_name+'-mean-error.png'
    plt.title('Predicted vs measured metabolome RMSE')
    plt.xlabel('Byproduct fraction, f')
    plt.ylabel('Cell line')
    if figsave_flag:
        figerror.savefig('../figures/'+file_name, dpi=300)
    if not disp_flag:
        plt.close(figerror)

    ##### Slope of regression
    figslope, ax = plt.subplots(figsize=(10, 10))
    slopes_df = pd.DataFrame(slopes.T, columns=f_arr, index=cell_line_names)
    sns.heatmap(data=slopes_df, linewidth=0.5, cmap='crest', ax=ax)
    file_name = str(k)+'-celltypes/'+net_state+fig_name+'-slopes.png'
    plt.title('Predicted vs measured metabolome slope')
    plt.xlabel('Byproduct fraction, f')
    plt.ylabel('Cell line')
    if figsave_flag:
        figslope.savefig('../figures/'+file_name, dpi=300)
    if not disp_flag:
        plt.close(figslope)

    ##### Intercept of regression
    figintcpt, ax = plt.subplots(figsize=(10, 10))
    int_df = pd.DataFrame(intercepts.T, columns=f_arr, index=cell_line_names)
    sns.heatmap(data=int_df, linewidth=0.5, cmap='crest', ax=ax)
    file_name = str(k)+'-celltypes/'+net_state+fig_name+'-intercepts.png'
    plt.title('Predicted vs measured metabolome intercept')
    plt.xlabel('Byproduct fraction, f')
    plt.ylabel('Cell line')
    if figsave_flag:
        figintcpt.savefig('../figures/'+file_name, dpi=300)
    if not disp_flag:
        plt.close(figintcpt)

def plot_networks(net, df_summary, which_net, n_reps, fig_path, figsave_flag):
    # net_plot = net_optim.copy()
    net_plot = net.copy()
    df_summary.loc[:, 'mean'] = df_summary.iloc[:, 1:n_reps].mean(1)

    celltype_labels = ['A', 'B', 'C', 'D', 'E']
    net_plot.iloc[:, 1] = [celltype_labels[i] for i in net.loc[:, 'celltypes'].values]

    net_plot.columns = np.array(['source', 'target', 'edgeType'])
    net_plot.loc[:, 'edge_attr'] = df_summary.loc[:, 'mean'].values
    net_plot = net_plot[net_plot.loc[:, 'edgeType'] != 0]
    net_temp = net_plot.copy()

    i_flip = np.where(net_temp.loc[:, 'edgeType'] == 3)[0]
    net_plot.iloc[i_flip, 0] = net_temp.iloc[i_flip, 1]
    net_plot.iloc[i_flip, 1] = net_temp.iloc[i_flip, 0]

    fig, ax = plt.subplots(1, 2, figsize=(12, 17))
    G_con = nx.from_pandas_edgelist(net_plot[net_plot.loc[:, 'edgeType']==2], source='source', target='target', edge_attr='edge_attr', create_using=nx.DiGraph)
    right, left = nx.bipartite.sets(G_con)

    nx.draw_networkx(G_con, arrows=True, pos=nx.bipartite_layout(G_con, left),
                    node_size=250, ax=ax[0])
    ax[0].set_title('Uptake links')

    G_pro = nx.from_pandas_edgelist(net_plot[net_plot.loc[:, 'edgeType']==3], source='source', target='target', edge_attr='edge_attr', create_using=nx.DiGraph)
    right, left = nx.bipartite.sets(G_pro)
    nx.draw_networkx(G_pro, arrows=True, pos=nx.bipartite_layout(G_pro, right),
                    node_size=250, ax=ax[1])
    ax[1].set_title('Secretion links')

    fig.tight_layout()

    if figsave_flag:
        fig.savefig(fig_path + '/' + which_net +'.png', dpi=300)
        plt.close(fig)