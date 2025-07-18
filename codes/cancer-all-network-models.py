"""
<!---------------------------
Name: cancer-recon-network
File: cancer-all-network-models
-----------------------------
Author: akshitgoyal, bvibishan
Data:   20/06/2025, 11:09:25
---------------------------->
"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy.stats import sem
from sklearn.linear_model import LinearRegression
from scipy.sparse import csr_matrix
import numpy.matlib
from scipy.optimize import minimize
from scipy.stats import pearsonr

import os

# %%
############ Figure size settings
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE, family='sans-serif', serif='Arial')          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('text')

from matplotlib.ticker import MaxNLocator
my_locator = MaxNLocator(6)

color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def figure_size_setting(WIDTH):
    #WIDTH = 700.0  # the number latex spits out
    FACTOR = 0.8  # the fraction of the width you'd like the figure to occupy
    fig_width_pt  = WIDTH * FACTOR
    inches_per_pt = 1.0 / 72.27
    golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
    fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio   # figure height in inches
    fig_dims    = [fig_width_in, fig_height_in] # fig dims as a list
    return fig_dims

# %%
######## Import the pickled file containing all processed data which are useful for simulations (the processing is
########### done in "cancer-data-processing.py")
import pickle
# pickle_in = open("cancer_network.pickle","rb")
all_networks, i_intake, names = pd.read_pickle('cancer_network.pickle')
# i_selfish = 0

# pickle_in = open("data.pickle","rb")
celltype_ID, celltypefreq, ec_metabolome_ID, ec_metabolome, met_baseline, core_mean = pd.read_pickle('data.pickle')

i_nonzero_celltypes = all_networks[0]['celltypes_ID'].unique()
i_nonzero_celltypes = np.sort(i_nonzero_celltypes)
i_nonzero_celltypes = celltype_ID.values.copy()
i_nonzero_metabolites = all_networks[0]['metabolites_ID'].unique()
# i_nonzero_metabolites = np.sort(i_nonzero_metabolites)

MAX_ID_celltypes = len(i_nonzero_celltypes)  # MAX_ID_celltypes is the maximum of ID labels for celltypes.
MAX_ID_metabolites = len(i_nonzero_metabolites)  # MAX_ID_metabolites is the maximum of ID labels for metabolites.

# %%
################ All functions
def get_network(net):

    i_nonzero_celltypes = net['celltypes_ID'].unique()
    i_nonzero_celltypes = np.sort(i_nonzero_celltypes)
    i_nonzero_celltypes = celltype_ID.values.copy()
    i_nonzero_metabolites = net['metabolites_ID'].unique()
    # i_nonzero_metabolites = np.sort(i_nonzero_metabolites)

    MAX_ID_celltypes = len(i_nonzero_celltypes)  # MAX_ID_celltypes is the maximum of ID labels for celltypes.
    MAX_ID_metabolites = len(i_nonzero_metabolites)  # MAX_ID_metabolites is the maximum of ID labels for metabolites.

    df_metabolites = pd.DataFrame.from_dict({'oldID': i_nonzero_metabolites, 'newID':list(range(len(i_nonzero_metabolites)))})
    df_metabolites.set_index('oldID', inplace=True)
    df_celltypes = pd.DataFrame.from_dict({'oldID': i_nonzero_celltypes, 'newID':list(range(len(i_nonzero_celltypes)))})
    df_celltypes.set_index('oldID', inplace=True)


    outgoingNodes = df_metabolites.reindex(net['metabolites_ID'].values).values.flatten()
    ingoingNodesTemp = df_celltypes.reindex(net['celltypes_ID'].values).values.flatten()
    edge_types = net.iloc[~np.isnan(ingoingNodesTemp),3].values
    outgoingNodes = outgoingNodes[~np.isnan(ingoingNodesTemp)]
    ingoingNodes = ingoingNodesTemp[~np.isnan(ingoingNodesTemp)].astype(int)

    net_reduced = pd.DataFrame.from_dict({'metabolites': outgoingNodes, 'celltypes':ingoingNodes, 'edgeType':edge_types})
    i_secretion = np.where(np.isin(net_reduced.iloc[:, -1], [3, 5]))[0]
    i_consumption = np.where(np.isin(net_reduced.iloc[:, -1], [3, 5]))[0]

    net_copy1 = net_reduced.copy()
    net_copy2 = net_reduced.copy()
    net_copy1.iloc[:, -1] = 0
    net_copy2.iloc[:, -1] = 0

    net_copy1.iloc[i_consumption, -1] = 2
    net_copy2.iloc[i_secretion, -1] = 3
    net_final = pd.concat([net_copy1, net_copy2])
    # net = net_reduced.copy()
    # net_temp = net.copy()
    # i_both = np.where(net_reduced['edgeType']==5)
    # net.iloc[i_both, -1] = 2#['edgeType'][net['edgeType']==5] = 2
    # net_temp.iloc[i_both, -1] = 3#['edgeType'][net_temp['edgeType']==5] = 3
    # net = pd.concat([net, net_temp]).drop_duplicates() #net.append(net_temp).drop_duplicates()
    # net_final = net.copy()

    return net_final, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites

def Ain_out(f, ct_hyp, diet, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
    '''
    Ain_out is a function used to create sparse matrices made of metabolites and celltypes 
    where metabolite consumption and production is considered. The matrices created are "m2b" and "b2m":
    (1) m2b is a matrix determines the nutrient splitting among celltypes, and
    (2) b2m is a matrix determines the byproducts generation.
    Both matrices have rows (columns?) representing bacterial species and columns (rows?) representing metabolites.
    Two matrices are created based on (1) the metabolite consumption and production network which is 
    encode in "net" as a dataframe, and (2) the hypothesised relative cell type frequencies "ct_hyp".
    '''
    valid_index = np.where((net['edgeType']==2) | (net['edgeType']==5))[0]
    row = net['metabolites'].iloc[valid_index]
    col = net['celltypes'].iloc[valid_index]
    data = np.ones((len(valid_index),))
    m2b = csr_matrix((data,(row,col)), shape=(MAX_ID_metabolites, MAX_ID_celltypes)).toarray()#.todense()
    in_degree = m2b.sum(0)

    valid_index = np.where((net['edgeType']==3) | (net['edgeType']==5))[0]
    row = net['metabolites'].iloc[valid_index]
    col = net['celltypes'].iloc[valid_index]
    data = np.ones((len(valid_index),))
    b2m = csr_matrix( (data,(row,col)), shape=(MAX_ID_metabolites, MAX_ID_celltypes)).toarray()#.todense()

    ########## Normalize the b2m by out_degree
    out_degree = b2m.sum(0).copy()
    out_degree[out_degree==0]=1e6
    b2m = b2m / out_degree

    ########## Normalize the m2b by proportion of microbial abundance in each individual
    ct_freq = ct_hyp/ct_hyp.sum()
    ct_hyp_repmat = numpy.matlib.repmat(ct_freq[np.newaxis,:], MAX_ID_metabolites, 1)
    m2b = m2b * ct_hyp_repmat # Uptake is proportional to cell type relative abundance
    # m2b = np.asarray([i*j for i, j in zip(m2b, diet.to_numpy(dtype=float))]) # Relative amount of nutrient taken up
    if in_degree_flag:
        m2b = m2b/in_degree
    
    m2b = np.float32(m2b)
    b2m = np.float32(b2m)

    ## Net secretion matrix for both cell types, following uptake and secretion of metabolites 
    m2m_total = np.zeros((MAX_ID_metabolites, MAX_ID_metabolites))
    m2m_total = np.array([i*j*k for i, j, k in zip(b2m, f*m2b, diet.values)])

    return [m2b, b2m, m2m_total]

def calc_metabolome(diet, m2b, m2m_total, numLevels_max, MAX_ID_metabolites):
    '''
    calc_metabolome is a function used to calculate the metabolome from the fitted nutrient intake from the
    model. It relies on (1) x: the nutrient intake, (2) i_intake: IDs of the nutrient intake, (3) m2m_layer: 
    a conversion matrix from the nutrient intake to the metabolite byproducts at a trophic level or layer, 
    and (4) numLevels_max: the number of trophic levels/layers in the model. The metabolome in the model is 
    assumed to be composed of two parts: (1) met_levels: all metabolites in the final trophic level/layer 
    (which is considered to be reaching the end of the gut because of the finite gut length and gut motility.),
    and (2) met_leftover_levels: all unusable metabolites from all previous trophic levels/layers. 
    '''
    met_levels = m2m_total.sum(1)
    met_leftover_levels = np.where(m2b.sum(1)==0, diet.to_numpy(), 0)

    metabolome_predicted = met_levels + met_leftover_levels
            
    return metabolome_predicted

def calc_pred_error(ct_hyp, net, numLevels_max, f, diet, ec_real, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
    '''
    pred_error is a function used to compute the logarithmic error between experimentally measured
    metagenome and predicted metagenome computed from the model for a certain nutrient intake. It relies on 
    (1) x: the nutrient intake, (2) i_intake: IDs of the nutrient intake, (3) m2b_total: a conversion matrix 
    from the nutrient intake to the total biomass, and (4) ct_hyp: hypothesised relative cell type frequenices. The 
    first three is used to compute the net gain in the intracellular metabolome predicted by the model "ic_pred" and compare it with the 
    experimentally measured net gain in intracellular metabolome "ic_real".
    '''

    m2b, b2m, m2m_total = Ain_out(f, ct_hyp, diet, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
    
    ec_pred = calc_metabolome(diet, m2b, m2m_total, numLevels_max, MAX_ID_metabolites) # Row sum of all secreted metabolites plus unused metabolites
    
    pred_error = (np.log10(ec_pred + 1e-10) - np.log10(ec_real + 1e-10)) / np.log10(ec_real +1e-10)
    pred_error = np.sqrt(np.dot(pred_error, pred_error.T)) #np.sqrt(np.sum(pred_error**2))
    return pred_error

def run_network_model(f, diet, col_name, k, cellnum_init, cellnum_max, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
    f = f * np.ones((1, MAX_ID_celltypes))
    numLevels_max = 1

    ######## Initial cell type frequencies and empirically measured extracellular metabolome for cell line given by col_name
    ct0 = cellnum_init*celltypefreq.to_numpy()
    ec_real = ec_metabolome[col_name].to_numpy(dtype=float)
 
    ct_full = np.zeros((MAX_ID_celltypes,)) # Final cell number, either fitted or taken depending on number of cell types

    ######## Compute matrices involving the metabolite consumption and generation:
    m2b, b2m, m2m_total = Ain_out(f, ct0, diet, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
    # m2m_total = m2b_multiple_levels(f, m2b, b2m, numLevels_max, MAX_ID_metabolites, MAX_ID_celltypes)

    ##### For k > 1, the model is converted into an optimization problem where the celltype frequencies are constantly changed to minimize the logarithmic error between experimentally measured metabolome and predicted metabolome computed from the model for a certain up-sec network and initial cell type distribution.
    if k > 1:
        my_args = (net, numLevels_max, f, diet, ec_real, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
        # fun = lambda ct: pred_error(ct, net, numLevels_max, f, x, ec_real, MAX_ID_metabolites, MAX_ID_celltypes)
        bnds = ((cellnum_init, cellnum_max), ) * len(ct0)
        constraint = {'type': 'eq', 'fun': lambda ct: ct.sum() - cellnum_max}
        res = minimize(calc_pred_error, ct0, args=my_args, method='trust-constr', bounds=bnds, options={'disp': False, 'maxiter': 1000}, tol=1e-3, constraints=constraint)
        ct_full = res.x #res.x.max()/cellnum_max
        
    #### For k = 1, no model fitting, final cell number is taken directly from cell number at confluency
    else:
        ct_full = cellnum_max

    metabolome_pred_unfilt = calc_metabolome(diet, m2b, m2m_total, numLevels_max, MAX_ID_metabolites)
    metabolome_measured_unfilt = ec_real.copy()

    ### Taking out unused metabolites from the analysis
    i_filter = np.where(m2b.sum(1)!=0)[0]
    metabolome_measured = metabolome_measured_unfilt[i_filter]
    metabolome_pred = metabolome_pred_unfilt[i_filter]

    ### Correlation between predicted and expected metabolome
    ec_corr = pearsonr(np.log10(metabolome_pred+1e-07), np.log10(metabolome_measured+1e-07))[0]
    # print('(Correlation coefficient, P-value) of predicted vs measured extracellular metabolome:')
    # print(pearsonr(np.log10(metabolome_pred+1e-07), np.log10(metabolome_measured+1e-07)))
    # print('-------------------------------------------------------------------------------------------------------------')

    ### Regression of predicted against expected metabolome
    model = LinearRegression()
    fit_obj = model.fit(np.log10(metabolome_pred+1e-07).reshape(-1, 1), np.log10(metabolome_measured+1e-07))
    slope = fit_obj.coef_[0]
    intercept = fit_obj.intercept_

    ### Mean squared error in metabolome prediction
    diff = np.log10(metabolome_pred+1e-07) - np.log10(metabolome_measured+1e-07)
    mean_error = np.sqrt(np.mean(diff**2))

    return [ec_corr, ct_full, mean_error, metabolome_pred, metabolome_measured, m2b, b2m]#, ec_corr_filtered, slope_filt, intercept_filt

def generate_random_network(net):
    rnet = net.copy()
    n_unused = (rnet.iloc[:, -1] == 0).sum() # Number of unused metabolites
    n_edges = rnet.shape[0]
    random_edges = np.random.choice([2, 3, 5], n_edges, replace=True) #Replace original nodes with some random choice of 2, 3 or 5
    random_i_unused = np.random.choice(np.arange(len(rnet)), n_unused, replace=False) # Randomly assigned unused status to the same number of metabolites as in the original network
    rnet.iloc[:, -1] = random_edges.copy()
    rnet.iloc[random_i_unused, -1] = 0
    return rnet
    # all_random_networks.append(rnet)

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

####### Error function for GutCP-based algorithm
def pred_error_addingLinks(x, m2b_ori, b2m_ori, net_ori, f, col_name, diet, in_degree_flag, cellnum_init, cellnum_max, i_nonzero_metabolites, i_nonzero_celltypes):
    '''
    pred_error_addingLinks is a function used to compute the logarithmic error between experimentally measured
    metagenome and predicted metagenome computed from the model for a certain nutrient intake. It relies on 
    (1) x: the nutrient intake, (2) i_intake: IDs of the nutrient intake, (3) m2b_total: a conversion matrix 
    from thenutrient intake to the total biomass, and (4) b_real: experimentally measured metagenome. The 
    first three is used to compute the metagenome predicted by the model "ba_pred" and compare it with the 
    experimentally measured metagenome "b_real".
    '''
    rmse_mets_dev = np.zeros((1))
    
    max_links = m2b_ori.shape[0] * m2b_ori.shape[1] # maximal number of links = number of specis * number of metabolites

    ######## Convert x to net structure (convert the adjacency matrix into the edge list):
    ### consumption links:
    x_consumption = x[:max_links]
    x_production = x[max_links:]

    a = net_ori.iloc[:max_links, 0].values
    b = net_ori.iloc[:max_links, 1].values
    c = np.where(x_consumption, 2, 0)
    net_added_consumption = pd.DataFrame({net_ori.columns[0]: a,
                                          net_ori.columns[1]: b,
                                          net_ori.columns[2]: c})
    
    a = net_ori.iloc[max_links:, 0].values
    b = net_ori.iloc[max_links:, 1].values
    c = np.where(x_production, 3, 0)
    net_added_production = pd.DataFrame({net_ori.columns[0]: a,
                                          net_ori.columns[1]: b,
                                          net_ori.columns[2]: c})
    # m2b_added = x[:max_links].reshape((m2b_ori.shape[0], m2b_ori.shape[1]))
    # i_add_consumption = np.where(m2b_added!=0)[0] # Indices of added links
    # # df_metabolites = pd.DataFrame.from_dict({'oldID': i_nonzero_metabolites, 'newID':list(range(len(i_nonzero_metabolites)))})

    # a = net_ori.iloc[:len(i_nonzero_metabolites), 0].values[i_add_consumption]
    # b = np.where(m2b_added==1)[1]#np.arange(len(i_nonzero_celltypes))[np.where(m2b_added >= thres)[1]]
    # c = [2] * len(b)
    # net_added_consumption = pd.DataFrame({net_ori.columns[0]:list(a), net_ori.columns[1]:list(b), net_ori.columns[2]:c})

    # ### production links:
    # b2m_added = x[max_links:].reshape((b2m_ori.shape[0], b2m_ori.shape[1]))
    # i_add_production = np.where(b2m_added!=0)[0]
    
    # a = net_ori.iloc[:len(i_nonzero_metabolites), 0].values[i_add_production]
    # b = np.where(b2m_added==1)[1]#np.arange(len(i_nonzero_celltypes))[np.where(b2m_added >= thres)[1]]
    # c = [3] * len(b)#np.where(b2m_added >= thres)[1].shape[0]
    # net_added_production = pd.DataFrame({net_ori.columns[0]:list(a), net_ori.columns[1]:list(b), net_ori.columns[2]:c})
    # ### new network with added links
    # # net = pd.concat([net_ori, net_added_consumption, net_added_production])

    # # ### Removing selected edges
    # # i_remove = np.where(m2b_added==0)[0]
    # # net_removed = net_ori.drop(net_ori.index[i_remove], inplace=False)

    net = pd.concat([net_added_consumption, net_added_production])
    n_changed = len(np.where(net.iloc[:, -1].values != net_ori.iloc[:, -1].values)[0])

    ec_corr, ct_full, mean_error, metabolome_pred, metabolome_measured, slope, intercept = run_network_model(f, diet, col_name, k, cellnum_init, cellnum_max, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

    # ct = ct_full*i_nonzero_celltypes
    ct_full = ct_full.reshape(celltypefreq.shape)
    m2b_final, b2m_final, m2m_total = Ain_out(f, ct_full, diet, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

    i_used_mets = np.where(m2b_final.sum(1)!=0)[0]
    # i_common = np.where(metabolome_measured * metabolome_pred != 0)[0]
    # metabolome_pred_common = metabolome_pred[i_common]
    # metabolome_measured_common = metabolome_measured[i_common]

    ######## compute the bias of the order of magnitude
    mets_dev = np.log10(metabolome_pred + 1e-7) - np.log10(metabolome_measured + 1e-7)
    # numMetabolites_list = i_used_mets.shape[0]
        
    if i_used_mets.shape[0] >= 2:
        rmse_mets_dev = mean_error

    else:
        var_expl = -1
        rmse_mets_dev = 7
    

    hyper_reg = 0.1
    pred_errorTotal = rmse_mets_dev + hyper_reg*n_changed #pred_error2 + hyper_reg * pred_error2 - (pred_error3 - 20) * 0.003 # with reward
    
    return [i_used_mets, pred_errorTotal, mets_dev, rmse_mets_dev]

def calculate_priors(bias_metabolome):
        # Prior probability is simply a rescaled value of the bias for each metabolite; cell type frequency not included here because there is no reference "measured" value unlike the metabolite levels
    consumption_prior, production_prior = np.zeros(MAX_ID_metabolites), np.zeros(MAX_ID_metabolites)
    i_con_prior = np.where(bias_metabolome > 0)[0]
    i_pro_prior = np.where(bias_metabolome < 0)[0]
    prior_temp = np.exp(np.abs(bias_metabolome))/np.exp(np.abs(bias_metabolome)).sum(0)
    consumption_prior[i_con_prior] = prior_temp[i_con_prior]
    production_prior[i_pro_prior] = prior_temp[i_pro_prior]
    prior_prob = np.concatenate([consumption_prior, production_prior]) # Appending the same array twice, first for consumption links and then for production links i.e., prior probability for a given metabolite depends on the error in prediction but is the same for consumption and production links
    prior_prob += 1/230 # All links have some constant probability to be chosen at random, independent of the bias in the metabolome prediction
    prior_prob /= prior_prob.sum(0) # Normalise to [0, 1]
    return prior_prob

def run_network_optimisation(f, cl, cellnum_init, cellnum_final, net_raw, diet, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
    error_list = []
    current_step_list = []
    pos_x_list = []
    metID_list = []
    celltypeID_list = []
    prior_list = []
    log_bias_list = []

    ct0 = cellnum_init*celltypefreq.to_numpy()

    # Save original network features and prediction errors
    net_ori, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(net_raw)

    m2b, b2m, m2m_total = Ain_out(f, ct0, diet, net_ori, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
    ####### Keep a record of the original network
    m2b_ori = (m2b!=0).astype(int).copy()
    b2m_ori = (b2m!=0).astype(int).copy()
    x_ori = np.concatenate([m2b_ori.flatten(), b2m_ori.flatten()]) # This is the adjacency matrix

    fun = lambda x: pred_error_addingLinks(x, m2b_ori, b2m_ori, net_ori, f, cl, diet, in_degree_flag, cellnum_init, cellnum_final, i_nonzero_metabolites, i_nonzero_celltypes)

    # Bias in the metabolome is calculated as the difference between predicted and measured metabolite levels; bias for metabolites not predicted by the model (pred=0) is set to -7 as an arbirarily large bias in prediction
    max_links = m2b.shape[0]*m2b.shape[1]
    i_used_mets, error_before, bias1, log_met_bias_init = fun(x_ori)
    bias_metabolome = np.zeros((m2b.shape[0],)) - 7
    bias_metabolome[i_used_mets, ] = bias1.copy()
    bias_metabolome_ori = bias_metabolome.copy()

    prior_prob = calculate_priors(bias_metabolome_ori)

    print('The original error is', error_before)
    error_list.append(error_before)

    x = x_ori.copy()

    # Network optimisation begins here
    inverseKT = 1.25
    Twindow = 500
    numStepsNotAdded = 0
    numAdditions = 0
    numDeletions = 0
    error_window = []
    for i in range(10000):
        # if i%50==0:
        #     print(i)
        if np.random.uniform(0,1,1)[0] <= 0.5: # Each step chooses randomly between adding a new link or removing an existing link
            i_x = np.random.choice(np.where(x==0)[0], 1, p=prior_prob[x==0]/np.sum(prior_prob[x==0]))[0] # New link is chosen based on the prior probability-smaller bias in prediction leads to smaller prior prob
            x[i_x] = 1
        else:
            # p_delete = 1/prior_prob[x==1]
            # p_delete = p_delete / np.sum(p_delete)
            i_x = np.random.choice(np.where(x==1)[0], 1, p=prior_prob[x==1]/np.sum(prior_prob[x==1]))[0] # New link is chosen based on the prior probability-smaller bias in prediction leads to smaller prior prob
            x[i_x] = 0
        i_used_mets, error_after, bias_metabolome, log_bias = fun(x) # Calculate prediction error with the modified network
        prior_prob = calculate_priors(bias_metabolome)

        if np.random.uniform(0,1,1)[0] < np.exp((inverseKT)*np.abs(error_after)): # If the reduction in error is large enough, the proposed link addition/removal is accepted
            error_before = error_after
            if x[i_x] == 1:
                # print('Addition accepted, error is ', error_before)
                numAdditions += 1
            else:
                # print('Deletion accepted, error is ', error_before)
                numDeletions += 1 
            error_list.append(error_before)
            current_step_list.append(i)
            pos_x_list.append(i_x) # Record every link whose status has changed in the adjacency matrix
            prior_list.append(prior_prob[i_x])
            log_bias_list.append(log_bias)
            if i_x < max_links:
                row_num = i_x // m2b_ori.shape[1]
                col_num = i_x - row_num * m2b_ori.shape[1]
            elif i_x >= max_links:
                i_x = i_x - m2b_ori.shape[0] * m2b_ori.shape[1]
                row_num = i_x // b2m_ori.shape[1]
                col_num = i_x - row_num * b2m_ori.shape[1]
            metID_list.append(row_num)
            celltypeID_list.append(col_num)
            numStepsNotAdded = 0
        else:  ## not accepted   
            x[i_x] = x_ori[i_x] # Maintain the original state
            numStepsNotAdded += 1
        error_window.append(error_before)
        if (i > Twindow) and ((error_window[-1] - error_window[-Twindow]) > -(np.sqrt(Twindow) / inverseKT)):
            break

    i_added = x[pos_x_list].astype(bool)
    i_deleted = ~x[pos_x_list].astype(bool)
    # i_used_mets, error_final, met_bias_final, log_met_bias_final = fun(x)

    # links_added, links_deleted = np.zeros(max_links*2), np.zeros(max_links*2)
    # links_added[np.array(pos_x_list)[i_added]] = 1
    # links_deleted[np.array(pos_x_list)[i_deleted]] = 1

    # consumption_added = links_added[:max_links]
    # production_added = links_added[max_links:]

    # consumption_deleted = links_deleted[:max_links]
    # production_deleted = links_deleted[max_links:]

    return [x_ori, x, error_list, i_added.sum(), i_deleted.sum(), log_bias_list]#, consumption_added, production_added, consumption_deleted, production_deleted]


# %%
###### Model initialisation
f_arr = np.linspace(0.1, 1, 5)
cell_line_names = core_mean.loc[:, 'Cell line'].to_numpy()
k = len(celltype_ID)  # Number of cell types in the model

######## Diet as the average of all the Baseline values
diet = met_baseline.mean(axis=1)

ec_corr = np.zeros((len(f_arr), len(cell_line_names)))
ct_full = np.zeros((len(f_arr), len(cell_line_names))) # For one cell type
mean_error = np.zeros((len(f_arr), len(cell_line_names)))
# ct_full = np.zeros((len(f_arr), len(cell_line_names), k)) # For more than one cell type
metabolome_pred = np.zeros((len(f_arr), len(cell_line_names), len(ec_metabolome)))
slopes = np.zeros_like(ec_corr)
intercepts = np.zeros_like(ec_corr)


 ## Source for initial and final cell numbers for the various cell lines: https://www.thermofisher.com/in/en/home/references/gibco-cell-culture-basics/cell-culture-protocols/cell-culture-useful-numbers.html
n_lines = len(core_mean.loc[:, 'Cell line'])
# nonadh_cell_lines = ['SR', 'MOLT-4', 'HL-60(TB)', 'K562', 'RPMI 8226', 'CCRF-CEM']
# i_nonadh = np.where(np.isin(core_mean.loc[:, 'Cell line'], nonadh_cell_lines))[0]

t75_cell_lines = ['NCI-H460', 'HCC-2998', 'SW620']
i_t75 = np.where(np.isin(cell_line_names, t75_cell_lines))[0]
cellnum_init_all = np.array([4.9e+06]*n_lines)
cellnum_init_all[i_t75] = 2.1e+06
cellnum_final_all = np.array([23.3e+06]*n_lines)
cellnum_final_all[i_t75] = 8.4e+06

in_degree_flag = False
if in_degree_flag:
    fig_name = 'with-in-degree'
else:
    fig_name = 'no-in-degree'

# %%
####### Run the model over the initialised parameters with the pickled initial network
"""No network optimisation simulations"""
# f_count = 0
# k = 3 # Number of cell types
net_state = 'stat-net/'

metabolome_pred = [[[]]]
metabolome_measured = [[[]]]

for i, f in enumerate(f_arr[:1]):
    pred_temp = [[]]
    measured_temp = [[]]
    for j, net in enumerate(all_networks):
        net_corrected, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(net)
        # f_count += 1
        ec_corr[i, j], ct_full[i, j], mean_error[i, j], pred, measured, slopes[i, j], intercepts[i, j] = run_network_model(f, diet, cell_line_names[j], k, cellnum_init_all[j], cellnum_final_all[j], net_corrected, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
        pred_temp.append(pred)
        measured_temp.append(measured)
    
    metabolome_pred.append(pred_temp[1:])
    metabolome_measured.append(measured_temp[1:])

metabolome_pred = metabolome_pred[1:]
metabolome_measured = metabolome_measured[1:]

# %%
######### All plots
figsave_flag = False
disp_flag = True

for i, f in enumerate(f_arr):
    plot_all_corrs(net_state, fig_name, k, f, metabolome_pred[i], metabolome_measured[i], ec_metabolome, figsave_flag, disp_flag)

plot_summary_stats(net_state, fig_name, k, f_arr, ec_corr, ct_full, mean_error, slopes, intercepts, figsave_flag, disp_flag)


# %%
####### Generate random networks, one for each cell line
# for i in range(len(all_networks)):
all_random_networks = []
for net in all_networks:
    rnet = generate_random_network(net)
    all_random_networks.append(rnet)


# %%
######## Run model with random initial networks
# f_count = 0
ec_corr = np.zeros((len(f_arr), len(cell_line_names)))
ct_full = np.zeros((len(f_arr), len(cell_line_names))) # For one cell type
mean_error = np.zeros((len(f_arr), len(cell_line_names)))
# ct_full = np.zeros((len(f_arr), len(cell_line_names), k)) # For more than one cell type
# metabolome_pred = np.zeros((len(f_arr), len(cell_line_names), len(ec_metabolome)))
metabolome_pred = [[[]]]
metabolome_measured = [[[]]]
slopes = np.zeros_like(ec_corr)
intercepts = np.zeros_like(ec_corr)

# k = 3 # Number of cell types
net_state = 'random-net/'

for i, f in enumerate(f_arr):
    pred_temp = [[]]
    measured_temp = [[]]
    for j, rnet in enumerate(all_random_networks):
        rnet_corrected, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(rnet)
        ec_corr[i, j], ct_full[i, j], mean_error[i, j], pred, measured, slopes[i, j], intercepts[i, j] = run_network_model(f, diet, cell_line_names[j], k, cellnum_init_all[j], cellnum_final_all[j], rnet_corrected, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
        pred_temp.append(pred)
        measured_temp.append(measured)

    metabolome_pred.append(pred_temp[1:])
    metabolome_measured.append(measured_temp[1:])

metabolome_pred = metabolome_pred[1:]
metabolome_measured = metabolome_measured[1:]

# %% 
########### All plots
figsave_flag = True
disp_flag = False

for i, f in enumerate(f_arr):
    plot_all_corrs(net_state, fig_name, k, f, metabolome_pred[i], metabolome_measured[i], ec_metabolome, figsave_flag, disp_flag)

plot_summary_stats(net_state, fig_name, k, f_arr, ec_corr, ct_full, mean_error, slopes, intercepts, figsave_flag, disp_flag)





# %%
"""Network optimisation simulations"""
############ Run network optimisation 'n_rep' times for a given cell line, each time starting with a new randomised network
n_reps = 5
cl = cell_line_names[0]
f = 0.5

cellnum_init = cellnum_init_all[0]
cellnum_final = cellnum_final_all[0]
x_optim_list = [[]]
x_ori_list = [[]]
error_list_all_reps = [[]]
log_bias_list = [[]]
# consumption_added = [[]]
# production_added = [[]]
# consumption_deleted = [[]]
# production_deleted = [[]]

in_degree_flag = False

for i in np.arange(n_reps):
    net_raw = generate_random_network(all_random_networks[0])
    x_ori, x, elist, n_added, n_deleted, log_bias = run_network_optimisation(f, cl, cellnum_init, cellnum_final, net_raw, diet, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

    x_ori_list.append(x_ori)
    x_optim_list.append(x)
    error_list_all_reps.append(elist)
    log_bias_list.append(log_bias)
    # consumption_added.append(con_add)
    # production_added.append(pro_add)
    # consumption_deleted.append(con_del)
    # production_deleted.append(pro_del)

    # n_added = con_add.sum() + pro_add.sum()
    # n_deleted = con_del.sum() + pro_del.sum()
    print('Round', i+1, 'of network optmisation ended with', n_added, 'links added and', n_deleted, 'links deleted.')
    print('------------------')

x_ori_list = np.array(x_ori_list[1:])
x_optim_list = np.array(x_optim_list[1:])
error_plot_list = np.array(error_list_all_reps[1:], dtype=object)
log_bias_list = np.array(log_bias_list[1:], dtype=object)
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

max_links = MAX_ID_metabolites*MAX_ID_celltypes
con_links_all = np.array([arr[:max_links] for arr in x_optim_list - x_ori_list])
pro_links_all = np.array([arr[max_links:] for arr in x_optim_list - x_ori_list])
df_con_links = pd.DataFrame(con_links_all.T)
df_pro_links = pd.DataFrame(pro_links_all.T)

df_con_links.loc[:, 'added'] = np.array([np.where(i==1)[0].shape[0] for i in con_links_all.T])
df_con_links.loc[:, 'removed'] = np.array([np.where(i==-1)[0].shape[0] for i in con_links_all.T])
df_pro_links.loc[:, 'added'] = np.array([np.where(i==1)[0].shape[0] for i in pro_links_all.T])
df_pro_links.loc[:, 'removed'] = np.array([np.where(i==-1)[0].shape[0] for i in pro_links_all.T])
df_con_links.reset_index(names='metabolite', inplace=True)
df_pro_links.reset_index(names='metabolite', inplace=True)

df_con_plot = df_con_links.melt(id_vars='metabolite', var_name = 'linkType', value_vars=['added', 'removed'], value_name='linkNumber', ignore_index=False)
df_pro_plot = df_pro_links.melt(id_vars='metabolite', var_name = 'linkType', value_vars=['added', 'removed'], value_name='linkNumber', ignore_index=False)


# %%
##### Visualising output
# net_state = 'optim-net/'
# fig_path = '../figures/'+str(k)+'-celltypes/'+net_state+'/'+cl
# try:
#     os.makedirs(fig_path)
# except:
#     pass

##### I'm calling this the general summary, whatever that means
f, ax = plt.subplots(2, 2, figsize=(7, 6))

#### A quick glance of where sims have begun and ended
for i in range(n_reps):
    sns.lineplot(error_plot_list[i], ax=ax[0, 0])
ax[0, 0].set_xlabel("Add/remove steps")
ax[0, 0].set_ylabel("Prediction error")
ax[0, 0].set_title("%d replicate runs" % n_reps)

#### How many steps of add/remove on average
sim_length = np.zeros(n_reps)
for i in range(n_reps):
    sim_length[i] = len(error_plot_list[i])
sns.histplot(sim_length, fill=True, element='step', 
             stat='density', alpha=0.25, bins=10, kde=True, ax=ax[0, 1])
ax[0, 1].set_xlabel('# steps')
ax[0, 1].set_title("Step number distribution")

#### Does it matter what the initial error is
initial_error = np.array([arr[0] for arr in error_plot_list])
final_error = np.array([arr[-1] for arr in error_plot_list])
sns.regplot(x=initial_error, y=final_error, color='k', ax = ax[1, 0],
            line_kws={'color': 'r', 'linewidth': 2}, scatter_kws={'s': 10})
rsq, pval = pearsonr(initial_error, final_error)
ax[1, 0].text(0.05, 0.9, f'$r^2$ = {rsq**2:.2f}, $p$ = {pval:.2f}', transform=ax[1, 0].transAxes)
# ax[1, 0].scatter(initial_error, final_error, c='k', s=5)
ax[1, 0].set_xlabel('Initial prediction error')
ax[1, 0].set_ylabel('Final prediction error')

for i in range(n_reps):
    sns.lineplot(log_bias_list[i], ax=ax[1, 1])
ax[1, 1].set_xlabel("Add/remove steps")
ax[1, 1].set_ylabel("Met_RMSE")
ax[1, 1].set_title("%d replicate runs" % n_reps)
# sns.regplot(x=initial_error, y=sim_length, color='k', ax = ax[1, 1],
#             line_kws={'color': 'r', 'linewidth': 2}, scatter_kws={'s': 10})
# rsq, pval = pearsonr(initial_error, sim_length)
# ax[1, 1].text(0.05, 0.9, f'$r^2$ = {rsq**2:.2f}, $p$ = {pval:.2f}', transform=ax[1, 1].transAxes)
# # ax[1, 0].scatter(initial_error, final_error, c='k', s=5)
# ax[1, 1].set_xlabel('Initial prediction error')
# ax[1, 1].set_ylabel('Step number')

f.suptitle('Network optimisation for %s' % cl)
f.tight_layout()
# f.savefig(fig_path+'/general-summary.png', dpi=300)

##### A visualisation of what is being added and removed on average, over 100 replicate runs
f, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(16, 8))
# ax = sns.barplot(data=df_added_long, x='metabolite', y='mean',
#                 hue='linkType', palette='crest', width=4)
sns.barplot(data=df_con_plot, x='metabolite', y='linkNumber', hue='linkType', palette='crest', ax=ax[0])
ax[0].set_title('Consumption links changed')
ax[0].set_ylabel('')

sns.barplot(data=df_pro_plot, x='metabolite', y='linkNumber', hue='linkType', palette='crest', ax=ax[1])
plt.tick_params(labelrotation=75)
ax[1].set_title('Production links changed')
ax[1].set_ylabel('')

f.supylabel('Changes per %d replicates' % n_reps)
f.tight_layout()

# # Extract bar coordinates for error bar placement
# x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
# y_coords = [p.get_height() for p in ax.patches]

# # Add custom error bars
# plt.errorbar(x=x_coords, y=y_coords, yerr=con_add_sd, fmt='none', c='black', capsize=3)

# plt.savefig(fig_path+'/added-secretion-links.png', dpi=300)
# %%
# #### Convert adjacency back to network topology
# a = np.array(metID_list)
# b = np.array(celltypeID_list)
# c = np.ones([len(metID_list)], dtype = int) * 3
# c[np.where(np.array(pos_x_list) < max_links)[0]] = 2
# net_modified = pd.DataFrame({net_ori.columns[0]:list(a), net_ori.columns[1]:list(b), net_ori.columns[2]:c})
# i_added = x[pos_x_list].astype(bool)
# i_deleted = ~x[pos_x_list].astype(bool)
# net_added = net_modified[i_added]
# net_deleted = net_modified[i_deleted]

# net_new = pd.concat([net_ori, net_added, net_deleted]).drop_duplicates(keep=False)

# print('The original network has',len(net_ori),'links')
# print('The new network has',len(net_new),'links')
# print('There are',len(net_added),'links added')
# print('There are',len(net_deleted),'links deleted')


# %%
##### Check out the best performing network
i_best_net = np.where(final_error == final_error.min())[0]
x_ori = x_ori_list[i_best_net].flatten()
x_optim = x_optim_list[i_best_net].flatten()


# %%
##### Sample consensus network from the above 100 runs
x1, x2 = np.zeros(MAX_ID_metabolites), np.zeros(MAX_ID_metabolites)
x1[np.where(np.array(consumption_added[1:]).sum(0) >= 10)[0]] = 1
x2[np.where(np.array(production_added[1:]).sum(0) >= 10)[0]] = 1

x1[np.where(np.array(consumption_deleted[1:]).sum(0) >= 10)[0]] = 0
x2[np.where(np.array(production_deleted[1:]).sum(0) >= 10)[0]] = 0

x_consensus = np.concatenate([x1, x2])

######## Convert x to net structure (convert the adjacency matrix into the edge list):
### consumption links:

max_links = MAX_ID_metabolites * MAX_ID_celltypes # maximal number of links = number of specis * number of metabolites
net_ori, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(all_random_networks[0])


m2b_added = x_optim[:max_links].reshape((MAX_ID_metabolites, MAX_ID_celltypes))
i_add_consumption = np.where(m2b_added!=0)[0] # Indices of added links
# df_metabolites = pd.DataFrame.from_dict({'oldID': i_nonzero_metabolites, 'newID':list(range(len(i_nonzero_metabolites)))})

a = net_ori.iloc[:len(i_nonzero_metabolites), 0].values[i_add_consumption]
b = np.where(m2b_added==1)[1]#np.arange(len(i_nonzero_celltypes))[np.where(m2b_added >= thres)[1]]
c = [2] * len(b)
c_index = net_ori.index[i_add_consumption]
net_added_consumption = pd.DataFrame({net_ori.columns[0]:list(a), net_ori.columns[1]:list(b), net_ori.columns[2]:c},
                                    index=c_index)

### production links:
b2m_added = x_optim[max_links:].reshape((MAX_ID_metabolites, MAX_ID_celltypes))
i_add_production = np.where(b2m_added!=0)[0]

a = net_ori.iloc[:len(i_nonzero_metabolites), 0].values[i_add_production]
b = np.where(b2m_added==1)[1]#np.arange(len(i_nonzero_celltypes))[np.where(b2m_added >= thres)[1]]
c = [3] * len(b)#np.where(b2m_added >= thres)[1].shape[0]
p_index = net_ori.index[i_add_production]
net_added_production = pd.DataFrame({net_ori.columns[0]:list(a), net_ori.columns[1]:list(b), net_ori.columns[2]:c},
                                    index=p_index)

net_optim = pd.concat([net_added_consumption, net_added_production])

ec_corr, ct_full, mean_error, metabolome_pred, metabolome_measured, slope, intercept = run_network_model(f_arr[0], diet, cell_line_names[0], k, cellnum_init_all[0], cellnum_final_all[0], net_optim, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)



# %%
######### Change in error with additions and deletions
# f = f_arr[0]
# ec_corr_old, ct_full_old, mean_error_old, metabolome_pred_old, metabolome_measured_old, slope_old, intercept_old = run_network_model(f, diet, cl, k, cellnum_init_all[0], cellnum_final_all[0], net_ori, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

# ec_corr, ct_full, mean_error, metabolome_pred, metabolome_measured, slope, intercept = run_network_model(f, diet, cl, k, cellnum_init_all[0], cellnum_final_all[0], net_new, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

# print('Metabolome deviation with old network is: ', mean_error_old)
# print('Metabolome deviation with improved network is: ', mean_error)
# print('------------------------------------------------------------------------')
# plt.figure()
# plt.plot(error_list, 'ko-')
# plt.ylabel('error')
# plt.xlabel('number of add/remove steps')

# fig, ax = plt.subplots(1, 2, sharey=True, figsize=(7, 4))
# ax[0].scatter(np.log10(metabolome_pred_old+1e-7), np.log10(metabolome_measured_old+1e-7), c='k', s=8)
# ax[0].plot([-7, 1], [-7, 1],'k-')
# ax[0].set_title('Old network')
# # ax[0].set_xlabel(r'$log_{10}\ Predicted\ metabolome$')
# ax[0].set_ylabel(r'$log_{10}\ Empirical\ data$')
# ax[1].scatter(np.log10(metabolome_pred+1e-7), np.log10(metabolome_measured+1e-7), c='k', s=8)
# ax[1].plot([-7, 1], [-7, 1], 'k-')
# ax[1].set_title('New network')
# # ax[1].set_xlabel(r'$log_{10}\ Predicted\ metabolome$')
# # ax[1].set_ylabel(r'$log_{10}\ Empirical\ data$')
# fig.supxlabel(r'$log_{10}\ Predicted\ metabolome$')
# plt.tight_layout()
# plt.show()
# # plt.scatter(np.log10(metabolome_pred+1e-7), np.log10(metabolome_measured+1e-7), c='k', s=4)
    
    