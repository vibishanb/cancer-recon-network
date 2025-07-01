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

    ########## Normalize the m2b by proportion of microbial abundance in each individual
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
    met_levels = m2m_total.sum(1)
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

    metabolome_pred_unfilt = calc_metabolome(x, m2b, m2m_total, numLevels_max, MAX_ID_metabolites)
    metabolome_measured_unfilt = ec_real.copy()

    ### Taking out unused metabolites from the analysis
    i_filter = np.where(m2b.sum(1)!=0)[0]
    metabolome_measured = metabolome_measured_unfilt[i_filter]
    metabolome_pred = metabolome_pred_unfilt[i_filter]

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
# f_count = 0
# k = 3 # Number of cell types
net_state = 'stat-net/'

metabolome_pred = [[[]]]
metabolome_measured = [[[]]]

for i, f in enumerate(f_arr):
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
figsave_flag = True
disp_flag = False

for i, f in enumerate(f_arr):
    plot_all_corrs(net_state, fig_name, k, f, metabolome_pred[i], metabolome_measured[i], ec_metabolome, figsave_flag, disp_flag)

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
####### Error function for GutCP-based algorithm
def pred_error_addingLinks(x, m2b_ori, b2m_ori, x_ori, net_ori, f, col_name, diet, cellnum_init, cellnum_max, i_nonzero_metabolites):
    '''
    pred_error_addingLinks is a function used to compute the logarithmic error between experimentally measured
    metagenome and predicted metagenome computed from the model for a certain nutrient intake. It relies on 
    (1) x: the nutrient intake, (2) i_intake: IDs of the nutrient intake, (3) m2b_total: a conversion matrix 
    from thenutrient intake to the total biomass, and (4) b_real: experimentally measured metagenome. The 
    first three is used to compute the metagenome predicted by the model "ba_pred" and compare it with the 
    experimentally measured metagenome "b_real".
    '''
    # f_byproduct = 0.9
    # f = f_byproduct * np.ones((MAX_ID_celltypes,1))

    # numLevels_max = 4

    corr_list_aveDiet = np.zeros((1))
    log_list_aveDiet = np.zeros((1))
    order_dev_list = np.zeros((1, MAX_ID_metabolites))
    # order_dev_metagenome_list = np.zeros((1, MAX_ID_celltypes))
    numMetabolites_list = np.zeros((1))
    # cell_line_names = ['U87MG', 'NSP']
    
    # for j, cl in enumerate(cell_line_names):
        ######### Add/remove links
    max_links = m2b_ori.shape[0] * m2b_ori.shape[1] # maximal number of links = number of specis * number of metabolites
    ######## Convert x to net structure (convert the adjacency matrix into the edge list):
    thres = 0.1
    x = x - x_ori
    ### consumption links:
    m2b_added = x[:max_links].reshape((m2b_ori.shape[0], m2b_ori.shape[1]))
    df_metabolites = pd.DataFrame.from_dict({'oldID': i_nonzero_metabolites, 'newID':list(range(len(i_nonzero_metabolites)))})

    a = df_metabolites['newID'].values[np.where(m2b_added >= thres)[0]]
    b = celltype_ID[np.where(m2b_added >= thres)[1]]
    c = [2] * np.where(m2b_added >= thres)[1].shape[0]
    net_added_consumption = pd.DataFrame({net_ori.columns[0]:list(a), net_ori.columns[1]:list(b), net_ori.columns[2]:c})
    ### production links:
    b2m_added = x[max_links:].reshape((m2b_ori.shape[0], m2b_ori.shape[1]))
    a = df_metabolites['newID'].values[np.where(b2m_added >= thres)[0]]
    b = celltype_ID[np.where(b2m_added >= thres)[1]]
    c = [3] * np.where(b2m_added >= thres)[1].shape[0]
    net_added_production = pd.DataFrame({net_ori.columns[0]:list(a), net_ori.columns[1]:list(b), net_ori.columns[2]:c})
    ### new network with added links
    net = pd.concat([net_ori, net_added_consumption, net_added_production])

    ### Removing random edges
    i_remove = np.random.choice(range(len(net_ori)), int(len(net_ori) * 0.2))
    net_removed = net.drop(net.index[i_remove], inplace=False)

    ec_corr, ct_full, metabolome_measured_common, metabolome_pred_common, i_common = run_network_model(f, col_name, net_removed, diet, cellnum_init, cellnum_max, MAX_ID_celltypes, MAX_ID_metabolites)
    # i_common = np.where(metabolome_measured * metabolome_pred > 1e-5)[0]

    ######## compute the bias of the order of magnitude
    order_dev_list[0, i_common] = np.log10(metabolome_measured_common + 1e-5) - np.log10(metabolome_pred_common + 1e-5)
    numMetabolites_list = i_common.shape[0]
        
    if i_common.shape[0] >= 2:
        corr_list_aveDiet = ec_corr
        log_list_aveDiet = np.mean((np.log10(metabolome_pred_common+1e-6) - np.log10(metabolome_measured_common+1e-6))**2)

    else:
        corr_list_aveDiet = -1
        log_list_aveDiet = 5
    
    pred_error1 = corr_list_aveDiet
    pred_error2 = np.sqrt(log_list_aveDiet)
    # pred_error2 = np.mean(log_list_aveDiet[:,1])
    # pred_error3 = np.sum(np.abs(x))
    pred_error3 = np.mean(numMetabolites_list)
    hyper_reg = 0.001
    pred_errorTotal = pred_error2 + hyper_reg * pred_error2 - (pred_error3 - 20) * 0.003 # with reward
    
    return [pred_errorTotal, np.abs(np.sum(order_dev_list, 0))]

# %%
##### Optimise networks using a GutCP-based algorithm-running for just one cell line first
cl = cell_line_names[0]
metabolome = ec_metabolome.loc[:, cl].to_numpy()
error_list = []
current_step_list = []
pos_x_list = []
metID_list = []
microbeID_list = []
prior_list = []
net_links_added = []

net, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(all_networks[0])

m2b, b2m = Ain_out(ct0, diet, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
######## Keep a record of the original network
m2b_ori = (m2b!=0).astype(int).copy()
b2m_ori = (b2m!=0).astype(int).copy()
x_ori = np.concatenate([m2b_ori.flatten(), b2m_ori.flatten()])
net_ori = net.copy()


fun = lambda x: pred_error_addingLinks(x, m2b_ori, b2m_ori, x_ori, net_ori, f, cl, diet, cellnum_init_all[0], cellnum_final_all[0], i_nonzero_metabolites)
max_links = m2b.shape[0]*m2b.shape[1]
x = x_ori.copy()
error_before, bias_metabolome = fun(x)
bias_metabolome_ori = bias_metabolome.copy()
# prior_prob1 = np.matlib.repmat(np.exp(3 * (bias_metagenome) / metabolome.shape[1])[np.newaxis, :], m2b.shape[0], 1) * np.matlib.repmat(np.exp(3 * bias_metabolome_ori / metabolome.shape[1])[:, np.newaxis], 1, m2b.shape[1])
prior_prob2 = np.matlib.repmat(np.exp(3 * bias_metabolome_ori / metabolome.shape[1])[:, np.newaxis], 1, m2b.shape[1])
prior_prob = prior_prob2.flatten() + 0.1 #np.concatenate([prior_prob1.flatten(), prior_prob2.flatten()]) + 0.1
prior_prob = prior_prob / np.sum(prior_prob)
print('The original error is', error_before)
error_list.append(error_before)

inverseKT = 5000
Twindow = 500
numStepsNotAdded = 0
error_window = []
for i in range(10000):
    if i%50==0:
        print(i)
    i_x = np.random.choice(np.where(x==0)[0], 1, p=prior_prob[x==0]/np.sum(prior_prob[x==0]))[0]
    x[i_x] = 1
    #x[i_x] = 1 - x[i_x]
    error_after, bias_metabolome = fun(x)
    ## Consumption links:
    # prior_prob1 = np.matlib.repmat(np.exp(3 * (bias_metagenome) / metabolome.shape[1])[np.newaxis, :], m2b.shape[0], 1) * np.matlib.repmat(np.exp(3 * bias_metabolome_ori / metabolome.shape[1])[:, np.newaxis], 1, m2b.shape[1])
    ## Production links:
    prior_prob2 = np.matlib.repmat(np.exp(3 * bias_metabolome_ori / metabolome.shape[1])[:, np.newaxis], 1, m2b.shape[1])
    prior_prob = prior_prob2.flatten() + 0.1#np.concatenate([prior_prob1.flatten(), prior_prob2.flatten()]) + 0.1
    prior_prob = prior_prob / np.sum(prior_prob)

    if np.random.uniform(0,1,1)[0] <= np.min([1, np.exp((inverseKT)*(error_before - error_after))]): 
        error_before = error_after
        print('accepted, error is', error_before)
        error_list.append(error_before)
        current_step_list.append(i)
        pos_x_list.append(i_x)
        prior_list.append(prior_prob[i_x])
        if i_x < max_links:
            row_num = i_x // m2b_ori.shape[1]
            col_num = i_x - row_num * m2b_ori.shape[1]
        elif i_x >= max_links:
            i_x = i_x - m2b_ori.shape[0] * m2b_ori.shape[1]
            row_num = i_x // b2m_ori.shape[1]
            col_num = i_x - row_num * b2m_ori.shape[1]
        metID_list.append(row_num)
        microbeID_list.append(col_num)
        numStepsNotAdded = 0
    else:  ## not accepted   
        x[i_x] = 0
        numStepsNotAdded += 1
        #x[i_x] = 1 - x[i_x]
    error_window.append(error_before)
    if (i > Twindow) and ((error_window[-1] - error_window[-Twindow]) > -(np.sqrt(Twindow) / inverseKT)):
        break
######## Convert x to net structure:
a = np.array(metID_list)
b = np.array(microbeID_list)
c = np.ones([len(metID_list)], dtype = int) * 3
c[np.where(np.array(pos_x_list) < max_links)[0]] = 2
net_added = pd.DataFrame({net_ori.columns[0]:list(a), net_ori.columns[1]:list(b), net_ori.columns[2]:c})
net_new = pd.concat([net_ori, net_added])

print('The original network has ',len(net_ori),'links')
print('The improved network has ',len(net_new),'links')

net_links_added.append(net_new)

######### Change in error with additions
plt.figure()
plt.plot(error_list)
plt.ylabel('error')
plt.xlabel('number of links added')
