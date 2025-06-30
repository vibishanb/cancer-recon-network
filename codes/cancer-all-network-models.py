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
pickle_in = open("cancer_network.pickle","rb")
all_networks, i_intake, names = pickle.load(pickle_in)
# i_selfish = 0

pickle_in = open("data.pickle","rb")
celltype_ID, celltypefreq, ec_metabolome_ID, ec_metabolome, met_baseline, core_mean = pickle.load(pickle_in)

# %%
################ All functions
def get_network(net):

    i_nonzero_celltypes = net['celltypes_ID'].unique()
    i_nonzero_celltypes = np.sort(i_nonzero_celltypes)
    i_nonzero_celltypes = celltype_ID.values.copy()
    i_nonzero_metabolites = net['metabolites_ID'].unique()
    i_nonzero_metabolites = np.sort(i_nonzero_metabolites)

    MAX_ID_celltypes = len(i_nonzero_celltypes)  # MAX_ID_celltypes is the maximum of ID labels for celltypes.
    MAX_ID_metabolites = len(i_nonzero_metabolites)  # MAX_ID_metabolites is the maximum of ID labels for metabolites.

    df_metabolites = pd.DataFrame.from_dict({'oldID': i_nonzero_metabolites, 'newID':list(range(len(i_nonzero_metabolites)))})
    df_metabolites.set_index('oldID', inplace=True)
    df_celltypes = pd.DataFrame.from_dict({'oldID': i_nonzero_celltypes, 'newID':list(range(len(i_nonzero_celltypes)))})
    df_celltypes.set_index('oldID', inplace=True)


    outgoingNodes = df_metabolites.reindex(net['metabolites_ID'].values).values.flatten()
    ingoingNodesTemp = df_celltypes.reindex(net['celltypes_ID'].values).values.flatten()
    edge_types = net.iloc[:, -1].to_numpy() #net.iloc[~np.isnan(ingoingNodesTemp),3].values
    outgoingNodes = outgoingNodes[~np.isnan(ingoingNodesTemp)]
    ingoingNodes = ingoingNodesTemp[~np.isnan(ingoingNodesTemp)].astype(int)

    net_reduced = pd.DataFrame.from_dict({'metabolites': outgoingNodes, 'celltypes':ingoingNodes, 'edgeType':edge_types})
    # net = net_reduced.copy()
    # net_temp = net.copy()
    # net.loc[net_reduced['edgeType']==5, 'edgeType'] = 2
    # net_temp.loc[net_reduced['edgeType']==5, 'edgeType'] = 3
    # net = pd.concat([net, net_temp]).drop_duplicates() #net.append(net_temp).drop_duplicates()
    # net_ori = net.copy()

    # celltype_ID_reduced = df_celltypes.reindex(celltype_ID).values.flatten()
    # celltype_ID_reduced = celltype_ID_reduced[~np.isnan(celltype_ID_reduced)].astype(int)

    # metabolome_ID_reduced = df_metabolites.reindex(ec_metabolome_ID).values.flatten()
    # metabolome_ID_reduced = metabolome_ID_reduced[~np.isnan(metabolome_ID_reduced)].astype(int)


    # # i_selfish_reduced = df_celltypes.reindex(i_selfish).values.flatten()
    # # i_selfish = i_selfish_reduced[~np.isnan(i_selfish_reduced)].astype(int)

    # i_intake_reduced = df_metabolites.loc[i_intake].values.flatten()
    # i_intake_reduced = i_intake_reduced[~np.isnan(i_intake_reduced)].astype(int)

    return net_reduced, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites

def Ain_out(ct_hyp, x, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
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
    # b2m = np.array([i * x.to_numpy(dtype=float) for i in b2m.T], dtype=float).T # Adding known external supply of metabolites to the uptake matrix

    ########## Normalize the m2b by proportion of microbial abundance in each individual
    # ct_hyp_repmat = numpy.matlib.repmat(ct_hyp[np.newaxis,:], MAX_ID_metabolites, 1)
    # m2b = m2b * ct_hyp_repmat
    # Also multiply by relative nutrient concentrations
    # in_degree = m2b.sum(1)
    # in_degree[in_degree==0]=1e6
    # m2b = m2b / numpy.matlib.repmat(in_degree[:,np.newaxis], 1, MAX_ID_celltypes)
    ct_freq = ct_hyp/ct_hyp.sum()
    ct_hyp_repmat = numpy.matlib.repmat(ct_freq[np.newaxis,:], MAX_ID_metabolites, 1)
    m2b = m2b * ct_hyp_repmat # Uptake is proportional to cell type relative abundance
    m2b = np.asarray([i*j for i, j in zip(m2b, x.to_numpy(dtype=float))]) # Relative amount of nutrient taken up
    if in_degree_flag:
        m2b = m2b/in_degree
    
    m2b = np.float32(m2b)
    b2m = np.float32(b2m)

    ## Net secretion matrix for both cell types, following uptake and secretion of metabolites 
    m2m_total = np.zeros((MAX_ID_metabolites, MAX_ID_metabolites))
    m2m_total = np.array([i*j for i, j in zip(b2m, f*m2b)])

    return [m2b, b2m, m2m_total]

def calc_metabolome(x, m2b, m2m_total, numLevels_max, MAX_ID_metabolites):
    '''
    calc_metabolome is a function used to calculate the metabolome from the fitted nutrient intake from the
    model. It relies on (1) x: the nutrient intake, (2) i_intake: IDs of the nutrient intake, (3) m2m_layer: 
    a conversion matrix from the nutrient intake to the metabolite byproducts at a trophic level or layer, 
    and (4) numLevels_max: the number of trophic levels/layers in the model. The metabolome in the model is 
    assumed to be composed of two parts: (1) met_levels: all metabolites in the final trophic level/layer 
    (which is considered to be reaching the end of the gut because of the finite gut length and gut motility.),
    and (2) met_leftover_levels: all unusable metabolites from all previous trophic levels/layers. 
    '''
    # i_x = x.index.to_numpy(dtype=int)
    # i_unused = np.where(m2b.sum(axis=1) == 0)[0]
    met_levels = m2m_total.sum(1)
    # met_leftover_levels = np.zeros_like(met_levels)
    # i_unused = np.where(m2b.sum(1)==0)[0]

    met_leftover_levels = np.where(m2b.sum(1)==0, x.to_numpy(), 0)

    metabolome_predicted = met_levels + met_leftover_levels
            
    return metabolome_predicted

def calc_pred_error(ct_hyp, net, numLevels_max, f, x, ec_real, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
    '''
    pred_error is a function used to compute the logarithmic error between experimentally measured
    metagenome and predicted metagenome computed from the model for a certain nutrient intake. It relies on 
    (1) x: the nutrient intake, (2) i_intake: IDs of the nutrient intake, (3) m2b_total: a conversion matrix 
    from the nutrient intake to the total biomass, and (4) ct_hyp: hypothesised relative cell type frequenices. The 
    first three is used to compute the net gain in the intracellular metabolome predicted by the model "ic_pred" and compare it with the 
    experimentally measured net gain in intracellular metabolome "ic_real".
    '''

    m2b, b2m, m2m_total = Ain_out(ct_hyp, x, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
    # m2m_total = m2b_multiple_levels(f, m2b, b2m, numLevels_max, MAX_ID_metabolites, MAX_ID_celltypes)
    
    ec_pred = calc_metabolome(x, m2b, m2m_total, numLevels_max, MAX_ID_metabolites) # Row sum of all secreted metabolites plus unused metabolites
    
    pred_error = (np.log10(ec_pred + 1e-10) - np.log10(ec_real + 1e-10)) / np.log10(ec_real +1e-10)
    pred_error = np.sqrt(np.dot(pred_error, pred_error.T)) #np.sqrt(np.sum(pred_error**2))
    return pred_error

def run_network_model(f, x, col_name, k, cellnum_init, cellnum_max, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
    f = f * np.ones((1, MAX_ID_celltypes))
    numLevels_max = 1

    ######## Initial cell type frequencies and empirically measured extracellular metabolome for cell line given by col_name
    ct0 = cellnum_init*celltypefreq.to_numpy()
    ec_real = ec_metabolome[col_name].to_numpy(dtype=float)
 
    ct_full = np.zeros((MAX_ID_celltypes,)) # Final cell number, either fitted or taken depending on number of cell types

    ######## Compute matrices involving the metabolite consumption and generation:
    m2b, b2m, m2m_total = Ain_out(ct0, x, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
    # m2m_total = m2b_multiple_levels(f, m2b, b2m, numLevels_max, MAX_ID_metabolites, MAX_ID_celltypes)

    ##### For k > 1, the model is converted into an optimization problem where the celltype frequencies are constantly changed to minimize the logarithmic error between experimentally measured metabolome and predicted metabolome computed from the model for a certain up-sec network and initial cell type distribution.
    if k > 1:
        my_args = (net, numLevels_max, f, x, ec_real, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
        # fun = lambda ct: pred_error(ct, net, numLevels_max, f, x, ec_real, MAX_ID_metabolites, MAX_ID_celltypes)
        bnds = ((cellnum_init, cellnum_max), ) * len(ct0)
        constraint = {'type': 'eq', 'fun': lambda ct: ct.sum() - cellnum_max}
        res = minimize(calc_pred_error, ct0, args=my_args, method='trust-constr', bounds=bnds, options={'disp': False, 'maxiter': 1000}, tol=1e-3, constraints=constraint)
        ct_full = res.x.max()/cellnum_max
        
    #### For k = 1, no model fitting, final cell number is taken directly from cell number at confluency
    else:
        ct_full = cellnum_max

    metabolome_pred = calc_metabolome(x, m2b, m2m_total, numLevels_max, MAX_ID_metabolites)
    metabolome_measured = ec_real.copy()

    ### Correlation between predicted and expected metabolome
    ec_corr = pearsonr(np.log10(metabolome_pred+1e-07), np.log10(metabolome_measured+1e-07))[0]
    print('(Correlation coefficient, P-value) of predicted vs measured extracellular metabolome:')
    print(pearsonr(np.log10(metabolome_pred+1e-07), np.log10(metabolome_measured+1e-07)))
    print('-------------------------------------------------------------------------------------------------------------')

    ### Regression of predicted against expected metabolome
    model = LinearRegression()
    fit_obj = model.fit(np.log10(metabolome_pred+1e-07).reshape(-1, 1), np.log10(metabolome_measured+1e-07))
    slope = fit_obj.coef_[0]
    intercept = fit_obj.intercept_

    ### Mean squared error in metabolome prediction
    diff = np.log10(metabolome_pred+1e-07) - np.log10(metabolome_measured+1e-07)
    mean_error = np.sqrt(np.mean(diff**2))

    # # Filtered predictions for random networks
    # i_filter = np.where(~((b2m == 0) + (m2b == 0)))[0]
    # metabolome_pred_filtered = metabolome_pred[i_filter]
    # metabolome_measured_filtered = metabolome_measured[i_filter]
    # ec_corr_filtered = pearsonr(np.log10(metabolome_pred_filtered+1e-07), np.log10(metabolome_measured_filtered+1e-07))[0]

    # model_filt = LinearRegression()
    # fit_obj_filt = model_filt.fit(np.log10(x_mod[i_filter]+1e-07), np.log10(metabolome_measured_filtered+1e-07))
    # slope_filt = fit_obj_filt.coef_
    # intercept_filt = fit_obj_filt.intercept_
    # y_pred_filt = slope_filt*np.log10(x_mod[i_filter]+1e-07) + intercept_filt
    # fig, ax = plt.subplots()
    # ax.plot(np.log10(x_mod[i_filter]+1e-07), np.log10(metabolome_measured_filtered+1e-07), 'ko')
    # ax.plot(np.log10(x_mod[i_filter]+1e-07), y_pred_filt,'k-')
    # ax.set_aspect('equal')
    # ax.set_xlabel('Diet')
    # ax.set_ylabel('Extracellular metabolome')
    # ax.set_title('Extracellular metabolome-filtered-%s-%s' %(f_name, col_name))

    return [ec_corr, ct_full, mean_error, metabolome_pred, metabolome_measured, slope, intercept]#, ec_corr_filtered, slope_filt, intercept_filt

def plot_all_corrs(net_state, fig_name, k, f, metabolome_pred, ec_metabolome, figsave_flag=False, disp_flag=True):

    try:
        os.makedirs('../figures/'+str(k)+'-celltypes/'+net_state)
    except:
        pass

    nrow = 10
    ncol = 6
    fig, ax = plt.subplots(nrow, ncol, figsize=(10, 10))
    cl_names = ec_metabolome.columns.to_numpy()
    count = 0
    for i in range(nrow):
        for j in range(ncol):
            if count < len(cl_names):
                ax[i, j].loglog(metabolome_pred[count, :], ec_metabolome.iloc[:, count], 'k.')
                ax[i, j].plot([1e-5, 1], [1e-5, 1],'k-')
                ax[i, j].set_title('%s' % cl_names[count], {'fontsize': 6}, pad=2)
                if i < nrow-1:
                    ax[i, j].tick_params(axis='x', which='both', labelbottom=False)
                if j > 0:
                    ax[i, j].tick_params(axis='y', which='both', labelleft=False)
                count += 1
                continue
            else:
                break
    fig.supxlabel('Predicted metabolome', y=0.05)
    fig.supylabel('Empirical data', x=0.05)
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
    plt.title('Predicted vs measured metabolome mean squared error')
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

# %%
###### Model initialisation
f_arr = np.linspace(0.1, 1, 5)
cell_line_names = core_mean.loc[:, 'Cell line'].to_numpy()
k = len(celltype_ID)  # Number of cell types in the model

######## Diet as the average of all the Baseline values
diet = met_baseline.mean(axis=1)

# ic_corr = np.zeros((len(f_arr), len(cell_line_names)))
ec_corr = np.zeros((len(f_arr), len(cell_line_names)))
ct_full = np.zeros((len(f_arr), len(cell_line_names))) # For one cell type
mean_error = np.zeros((len(f_arr), len(cell_line_names)))
# ct_full = np.zeros((len(f_arr), len(cell_line_names), k)) # For more than one cell type
metabolome_pred = np.zeros((len(f_arr), len(cell_line_names), len(ec_metabolome)))
slopes = np.zeros_like(ec_corr)
intercepts = np.zeros_like(ec_corr)
# ec_corr_filt = np.zeros_like(ec_corr)
# slopes_filt = np.zeros_like(ec_corr)
# intercepts_filt = np.zeros_like(ec_corr)

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
# f_count = 0
# k = 3 # Number of cell types
net_state = 'stat-net/'

for i, f in enumerate(f_arr):
    for j, net in enumerate(all_networks):
        net_corrected, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(net)
        # f_count += 1
        ec_corr[i, j], ct_full[i, j], mean_error[i, j], metabolome_pred[i, j], metabolome_measured, slopes[i, j], intercepts[i, j] = run_network_model(f, diet, cell_line_names[j], k, cellnum_init_all[j], cellnum_final_all[j], net_corrected, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

# %%
######### All plots
figsave_flag = True
disp_flag = False

for i, f in enumerate(f_arr):
    plot_all_corrs(net_state, fig_name, k, f, metabolome_pred[i, :, :], ec_metabolome, figsave_flag, disp_flag)

plot_summary_stats(net_state, fig_name, k, f_arr, ec_corr, ct_full, mean_error, slopes, intercepts, figsave_flag, disp_flag)

# %%
####### Generate random networks, one for each cell line
all_random_networks = []
for i in range(len(all_networks)):
    rnet = all_networks[i].copy()
    n_unused = (rnet.iloc[:, -1] == 0).sum() # Number of unused metabolites
    n_edges = rnet.shape[0]
    random_edges = np.random.choice([2, 3, 5], n_edges, replace=True) #Replace original nodes with some random choice of 2, 3 or 5
    random_i_unused = np.random.choice(rnet.index, n_unused, replace=False) # Randomly assigned unused status to the same number of metabolites as in the original network
    rnet.iloc[:, -1] = random_edges.copy()
    rnet.iloc[random_i_unused, -1] = 0
    all_random_networks.append(rnet)

# %%
######## Run model with random initial networks
# f_count = 0
ec_corr = np.zeros((len(f_arr), len(cell_line_names)))
ct_full = np.zeros((len(f_arr), len(cell_line_names))) # For one cell type
mean_error = np.zeros((len(f_arr), len(cell_line_names)))
# ct_full = np.zeros((len(f_arr), len(cell_line_names), k)) # For more than one cell type
metabolome_pred = np.zeros((len(f_arr), len(cell_line_names), len(ec_metabolome)))
slopes = np.zeros_like(ec_corr)
intercepts = np.zeros_like(ec_corr)

# k = 3 # Number of cell types
net_state = 'random-net/'

for i, f in enumerate(f_arr[:1]):
    for j, rnet in enumerate(all_random_networks[:1]):
        rnet_corrected, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(rnet)
        ec_corr[i, j], ct_full[i, j], mean_error[i, j], metabolome_pred[i, j], metabolome_measured, slopes[i, j], intercepts[i, j] = run_network_model(f, diet, cell_line_names[j], k, cellnum_init_all[j], cellnum_final_all[j], rnet_corrected, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

# %% 
########### All plots
figsave_flag = False
disp_flag = True

for i, f in enumerate(f_arr[:1]):
    plot_all_corrs(net_state, fig_name, k, f, metabolome_pred[i, :, :], ec_metabolome, figsave_flag, disp_flag)

plot_summary_stats(net_state, fig_name, k, f_arr, ec_corr, ct_full, mean_error, slopes, intercepts, figsave_flag, disp_flag)

# %%
