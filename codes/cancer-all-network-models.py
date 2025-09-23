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
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from scipy.stats import pearsonr
import networkx as nx
from multiprocessing import Pool

import os
import numpy.matlib
import all_plots

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
    i_consumption = np.where(np.isin(net_reduced.iloc[:, -1], [2, 5]))[0]

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

def calculate_metabolome_from_net(f, ct_hyp, diet, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
    '''
    This is a function used to create sparse matrices made of metabolites and celltypes 
    where metabolite consumption and production is considered. The matrices created are "m2b" and "b2m":
    (1) m2b is a matrix determines nutrient uptake by celltypes, and
    (2) b2m is a matrix determines the nutrient secretion.
    Both matrices have rows representing cancer celltypes and columns representing metabolites.
    Two matrices are created based on (1) the metabolite consumption and production network which is 
    encode in "net" as a dataframe, and (2) the hypothesised relative cell type frequencies "ct_hyp". Those metabolites that are not taken up by any celltype are inherited as is from the diet supplied, while uptake of metabolites is assumed to be complete i.e., even if only a single celltype can take up a metabolite, it consumes all of it from the environment.
    '''
    #### A^in
    valid_index = np.where((net['edgeType']==2) | (net['edgeType']==5))[0]
    col = net['metabolites'].iloc[valid_index]
    row = net['celltypes'].iloc[valid_index]
    data = np.ones((len(valid_index),))
    m2b = csr_matrix((data,(row,col)), shape=(MAX_ID_celltypes, MAX_ID_metabolites)).toarray()#.todense()
    in_degree = m2b.sum(0)

    #### A^out
    valid_index = np.where((net['edgeType']==3) | (net['edgeType']==5))[0]
    col = net['metabolites'].iloc[valid_index]
    row = net['celltypes'].iloc[valid_index]
    data = np.ones((len(valid_index),))
    b2m = csr_matrix( (data,(row,col)), shape=(MAX_ID_celltypes, MAX_ID_metabolites)).toarray()#.todense()

    if in_degree_flag:
        m2b = m2b/in_degree
    
    ##### Intake matrix calculation
    cellnum = np.matlib.repmat(ct_hyp[:, np.newaxis], 1, MAX_ID_metabolites) 
    con_matrix = m2b*cellnum
    ### Normalising consumption of metabolites by the total number of consumers of each metabolite
    total_cellnum = con_matrix.sum(0)
    tau = np.zeros(len(total_cellnum))
    i_nonzero = np.where(total_cellnum > 0)[0]
    tau[i_nonzero] += 1/total_cellnum[i_nonzero]
    con_norm = con_matrix * tau
    intake_vector = np.dot(con_norm, diet.values)

    ##### Output matrix
    out_degree = b2m.sum(1)
    inv_out_degree = np.zeros(len(out_degree))
    i_nonzero = np.where(out_degree > 0)[0]
    inv_out_degree[i_nonzero] += f/out_degree[i_nonzero] # Secretion by each celltype is split equally between all the metabolites it secretes
    out_mult = np.matlib.repmat(inv_out_degree[:, np.newaxis], 1, MAX_ID_metabolites)
    output_matrix = b2m * out_mult

    ###### Final metabolome-secreted + unused
    secreted_metabolome = np.dot(output_matrix.T, intake_vector)
    i_unused = np.where(in_degree == 0)[0] # Metabolites not consumed by any celltype
    metabolome_unused = np.zeros(len(diet.values))
    metabolome_unused[i_unused] += diet.values[i_unused]

    metabolome_pred = secreted_metabolome + metabolome_unused

    return [m2b, b2m, metabolome_pred]

def calc_pred_error(ct_hyp, net, f, diet, ec_real, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
    '''
    pred_error is a function used to compute the logarithmic error between experimentally measured
    metagenome and predicted metagenome computed from the model for a certain nutrient intake. It relies on 
    (1) diet: the nutrient intake, (2) i_intake: IDs of the nutrient intake, (3) m2b_total: a conversion matrix 
    from the nutrient intake to the total biomass, and (4) ct_hyp: hypothesised relative cell type frequenices. The 
    first three is used to compute the net gain in the intracellular metabolome predicted by the model "ic_pred" and compare it with the 
    experimentally measured net gain in intracellular metabolome "ic_real".
    '''

    m2b, b2m, ec_pred = calculate_metabolome_from_net(f, ct_hyp, diet, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
    

    i_nonzero = np.where((ec_pred * ec_real) > 0)[0]
    diff = np.log10(ec_pred[i_nonzero]) - np.log10(ec_real[i_nonzero]) # / np.log10(ec_real[i_nonzero])
    pred_error = np.sqrt(np.mean(diff**2)) #np.sqrt(np.dot(pred_error, pred_error.T)) #np.sqrt(np.sum(pred_error**2))
    return pred_error

def run_network_model(f, diet, col_name, cellnum_init, cellnum_max, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
    ######## Initial cell type frequencies and empirically measured extracellular metabolome for cell line given by col_name
    ct0 = cellnum_init*celltypefreq.values
    ec_real = ec_metabolome[col_name].values

    ct_final = np.zeros_like(ct0) # Final cell number, either fitted or taken depending on number of cell types

    ##### For max celltypes > 1, the model is converted into an optimization problem where the celltype frequencies are learned to minimize the logarithmic error between experimentally measured metabolome and predicted metabolome computed from the model for a certain up-sec network and initial cell type distribution.
    if MAX_ID_celltypes > 1:
        my_args = (net, f, diet, ec_real, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
        bnds = ((cellnum_init, cellnum_max), ) * len(ct0)
        constraint = {'type': 'eq', 'fun': lambda ct: ct.sum() - cellnum_max}
        res = minimize(calc_pred_error, ct0, args=my_args, method='SLSQP', bounds=bnds, options={'disp': False, 'maxiter': 1000}, tol=1e-3, constraints=constraint)
        ct_final = res.x #res.x.max()/cellnum_max
        
    #### For max celltypes = 1, no model fitting, final cell number is taken directly from cell number at confluency
    else:
        ct_final = cellnum_max
    
    ######## Compute matrices and predicted metabolome from learnt celltype abundances
    m2b, b2m, metabolome_pred_unfilt = calculate_metabolome_from_net(f, ct_final, diet, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
    metabolome_measured_unfilt = ec_real.copy()

    ### Taking out unused metabolites from the analysis
    i_nonzero = np.where((metabolome_pred_unfilt * metabolome_measured_unfilt) > 0)[0]
    metabolome_measured = metabolome_measured_unfilt[i_nonzero]
    metabolome_pred = metabolome_pred_unfilt[i_nonzero]

    ### Correlation between predicted and expected metabolome
    ec_corr = pearsonr(np.log10(metabolome_pred), np.log10(metabolome_measured))[0]

    ### Mean squared error in metabolome prediction
    diff = np.log10(metabolome_pred) - np.log10(metabolome_measured)
    mean_error = np.sqrt(np.mean(diff**2))

    return [ec_corr, ct_final, mean_error, metabolome_pred, metabolome_measured]#, slope_filt, intercept_filt

def generate_random_network(net, bias_metabolome):
    rnet = net.copy()
    # n_unused = (rnet.iloc[:, -1] == 0).sum() # Number of unused metabolites
    # n_edges = rnet.shape[0]
    i_consumption_edges = np.where(bias_metabolome > 0)[0]
    i_production_edges = np.where(bias_metabolome < 0)[0]

    con_edges = np.random.choice([0, 2], len(i_consumption_edges), replace=True)
    pro_edges = np.random.choice([0, 3], len(i_production_edges), replace=True)
    rnet.iloc[i_consumption_edges, -1] = con_edges.copy()
    rnet.iloc[i_production_edges, -1] = pro_edges.copy()

    if MAX_ID_celltypes > 1:    
        assigned_edges = rnet.iloc[:len(bias_metabolome), -1].values
        shuffled_edges = np.repeat(assigned_edges, MAX_ID_celltypes-1)
        shuffled_edges = np.random.permutation(shuffled_edges)
        
        rnet.iloc[len(bias_metabolome):, -1] = shuffled_edges.copy()
    # random_edges = np.random.choice([2, 3, 5], n_edges, replace=True) #Replace original nodes with some random choice of 2, 3 or 5
    # random_i_unused = np.random.choice(np.arange(len(rnet)), n_unused, replace=False) # Randomly assigned unused status to the same number of metabolites as in the original network
    # rnet.iloc[:, -1] = random_edges.copy()
    # rnet.iloc[random_i_unused, -1] = 0
    return rnet
    # all_random_networks.append(rnet)

####### Error function for GutCP-based algorithm
def pred_error_addingLinks(x, net_ori, f, col_name, diet, in_degree_flag, cellnum_init, cellnum_max, pred_params):
    '''
    pred_error_addingLinks is a function used to compute the rms error between experimentally measured
    metagenome and predicted metagenome computed from the model for a certain nutrient intake. It relies on 
    (1) x: the nutrient intake, (2) i_intake: IDs of the nutrient intake, (3) m2b_total: a conversion matrix 
    from thenutrient intake to the total biomass, and (4) b_real: experimentally measured metagenome. The 
    first three is used to compute the metagenome predicted by the model "ba_pred" and compare it with the 
    experimentally measured metagenome "b_real".
    '''
    # rmse_mets_dev = np.zeros((1))
    
    max_links = MAX_ID_celltypes * MAX_ID_metabolites # maximal number of links = number of celltypes * number of metabolites

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

    net = pd.concat([net_added_consumption, net_added_production])
    n_changed = len(np.where(net.iloc[:, -1].values != net_ori.iloc[:, -1].values)[0])

    ec_corr, ct_full, mean_error, metabolome_pred, metabolome_measured = run_network_model(f, diet, col_name, cellnum_init, cellnum_max, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

    n_predicted = len(np.where(metabolome_pred > 0)[0])

    ######## compute the bias of the order of magnitude
    mets_dev = np.log10(metabolome_pred) - np.log10(metabolome_measured)
    # numMetabolites_list = i_used_mets.shape[0]
        
    # if i_used_mets.shape[0] >= 2:
    #     rmse_mets_dev = mean_error

    # else:
    #     # var_expl = -1
    #     rmse_mets_dev = 7
    
    penalty_param = pred_params['penalty']
    reward_param = pred_params['reward']
    # hyper_reg = 0.00001
    pred_errorTotal = mean_error + penalty_param*n_changed - reward_param*n_predicted #+ hyper_reg*n_changed #pred_error2 + hyper_reg * pred_error2 - (pred_error3 - 20) * 0.003 # with reward
    
    return [pred_errorTotal, metabolome_pred, metabolome_measured, mets_dev, mean_error]

def calculate_priors(bias_metabolome):
        # Prior probability is simply a rescaled value of the bias for each metabolite; cell type frequency not included here because there is no reference "measured" value unlike the metabolite levels
    consumption_prior, production_prior = np.zeros(MAX_ID_metabolites), np.zeros(MAX_ID_metabolites)
    i_con_prior = np.where(bias_metabolome > 0)[0]
    i_pro_prior = np.where(bias_metabolome < 0)[0]
    prior_temp = np.exp(np.abs(bias_metabolome))/np.exp(np.abs(bias_metabolome)).sum(0)
    consumption_prior[i_con_prior] = prior_temp[i_con_prior]
    production_prior[i_pro_prior] = prior_temp[i_pro_prior]
    consumption_prior = np.repeat(consumption_prior, MAX_ID_celltypes)
    production_prior = np.repeat(production_prior, MAX_ID_celltypes)
    prior_prob = np.concatenate([consumption_prior, production_prior]) # Appending the same array twice, first for consumption links and then for production links i.e., prior probability for a given metabolite depends on the error in prediction but is the same for consumption and production links
    prior_prob += 1/230 # All links have some constant probability to be chosen at random, independent of the bias in the metabolome prediction
    prior_prob /= prior_prob.sum(0) # Normalise to [0, 1]
    return prior_prob

def run_network_optimisation(all_params):
#(f, cl, cellnum_init, cellnum_final, net_raw, diet, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
    f = all_params[0]
    cl = all_params[1]
    cellnum_init = all_params[2]
    cellnum_final = all_params[3]
    net_raw = all_params[4]
    diet = all_params[5]
    in_degree_flag = all_params[6]
    MAX_ID_metabolites = all_params[7]
    MAX_ID_celltypes = all_params[8]
    
    error_list = []
    current_step_list = []
    pos_x_list = []
    metID_list = []
    celltypeID_list = []
    prior_list = []
    log_bias_list = []
    metabolome_pred_list = []

    ct0 = cellnum_init*celltypefreq.to_numpy()

    # Save original network features and prediction errors
    net_ori, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(net_raw)

    m2b, b2m, met_pred = calculate_metabolome_from_net(f, ct0, diet, net_ori, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
    ####### Keep a record of the original network
    m2b_ori = (m2b!=0).astype(int).copy()
    b2m_ori = (b2m!=0).astype(int).copy()
    x_ori = np.concatenate([m2b_ori.flatten(), b2m_ori.flatten()]) # This is the adjacency matrix

    # fun = lambda x: pred_error_addingLinks(x, net_ori, f, cl, diet, in_degree_flag, cellnum_init, cellnum_final)

    # Bias in the metabolome is calculated as the difference between predicted and measured metabolite levels
    max_links = m2b.shape[0]*m2b.shape[1]
    pred_params = {'penalty': all_params[10], 'reward': all_params[11]}

    error_before, metabolome_pred_before, metabolome_meas_before, bias_metabolome, log_met_bias_init = pred_error_addingLinks(x_ori, net_ori, f, cl, diet, in_degree_flag, cellnum_init, cellnum_final, pred_params)
    bias_metabolome_ori = bias_metabolome.copy()

    prior_prob = calculate_priors(bias_metabolome_ori)

    # print('The original error is', error_before)
    error_list.append(error_before)

    x = x_ori.copy()

    # Network optimisation begins here
    kT = all_params[9]
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
            i_x = np.random.choice(np.where(x==1)[0], 1, p=prior_prob[x==1]/np.sum(prior_prob[x==1]))[0] # New link is chosen based on the prior probability-smaller bias in prediction leads to smaller prior prob
            x[i_x] = 0

        error_after, metabolome_pred_after, metabolome_meas_after, bias_metabolome, log_bias = pred_error_addingLinks(x, net_ori, f, cl, diet, in_degree_flag, cellnum_init, cellnum_final, pred_params) # Calculate prediction error with the modified network
        prior_prob = calculate_priors(bias_metabolome)

        if np.random.uniform(0,1,1)[0] < np.min([1, np.exp((error_before - error_after)/kT)]): # If the reduction in error is large enough, the proposed link addition/removal is accepted
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
            metabolome_pred_list.append(metabolome_pred_after)
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
        if (i > Twindow) and ((error_window[-1] - error_window[-Twindow]) > -(np.sqrt(Twindow)*kT)):
            break

    return log_bias_list[-1] #[x_ori, x, error_list, numAdditions, numDeletions, metabolome_pred_before, metabolome_meas_before,  metabolome_pred_after, metabolome_meas_after, log_bias_list]#, consumption_added, production_added, consumption_deleted, production_deleted]



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

####### Generate random networks, one for each cell line
# for i in range(len(all_networks)):
all_random_networks = []
for i, net in enumerate(all_networks):
    bias_metabolome = ec_metabolome.iloc[:, i].values - diet.values
    rnet = generate_random_network(net, bias_metabolome)
    all_random_networks.append(rnet)


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
metabolome_pred_before_list = [[]]
metabolome_meas_before_list = [[]]
metabolome_pred_after_list = [[]]
metabolome_meas_after_list = [[]]

in_degree_flag = False

for i in np.arange(n_reps):
    bias = ec_metabolome.iloc[:, 0].values - diet.values
    net_raw = generate_random_network(all_random_networks[0], bias)
    x_ori, x, elist, n_added, n_deleted, met_pred_before, met_meas_before, met_pred_after, met_meas_after, log_bias = run_network_optimisation(f, cl, cellnum_init, cellnum_final, net_raw, diet, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

    x_ori_list.append(x_ori)
    x_optim_list.append(x)
    error_list_all_reps.append(elist)
    log_bias_list.append(log_bias)
    metabolome_pred_before_list.append(met_pred_before)
    metabolome_pred_after_list.append(met_pred_after)
    metabolome_meas_before_list.append(met_meas_before)
    metabolome_meas_after_list.append(met_meas_after)

    print('Round', i+1, 'of network optmisation ended with', n_added, 'links added and', n_deleted, 'links deleted.')
    print('The final error is', elist[-1])
    print('------------------')

x_ori_list = np.array(x_ori_list[1:])
x_optim_list = np.array(x_optim_list[1:])
error_plot_list = np.array(error_list_all_reps[1:], dtype=object)
log_bias_list = np.array(log_bias_list[1:], dtype=object)
metabolome_pred_before_list = np.array(metabolome_pred_before_list[1:], dtype=object)
metabolome_pred_after_list = np.array(metabolome_pred_after_list[1:], dtype=object)
metabolome_meas_before_list = np.array(metabolome_meas_before_list[1:], dtype=object)
metabolome_meas_after_list = np.array(metabolome_meas_after_list[1:], dtype=object)

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
net_state = 'optim-net/'
figsave_flag = False
fig_path = '../figures/'+str(k)+'-celltypes/'+net_state+cl
try:
    os.makedirs(fig_path)
except:
    pass

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
if figsave_flag:
    f.savefig(fig_path+'/no-penalties-all-replicates-summary.png', dpi=300)
    plt.close(f)


# %%
##### Check out the best performing network
# pred_error_change = np.array([list[0]-list[-1] for list in log_bias_list])
final_pred_error = np.array([arr[-1] for arr in log_bias_list])

i_best_net = np.where(final_pred_error == final_pred_error.min())[0]#np.where(pred_error_change == pred_error_change.max())[0]
x_ori_best = x_ori_list[i_best_net].flatten()
x_optim_best = x_optim_list[i_best_net].flatten()

met_pred_before = metabolome_pred_before_list[i_best_net][0]
met_pred_after = metabolome_pred_after_list[i_best_net][0]

met_meas_before = metabolome_meas_before_list[i_best_net][0]
met_meas_after = metabolome_meas_after_list[i_best_net][0]


# %%
######## Convert x to net structure (convert the adjacency matrix into the edge list)

max_links = MAX_ID_metabolites * MAX_ID_celltypes # maximal number of links = number of specis * number of metabolites
net_temp, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(net_raw)

######## Original network
x_consumption = x_ori_best[:max_links]
x_production = x_ori_best[max_links:]

a = net_temp.iloc[:max_links, 0].values
b = net_temp.iloc[:max_links, 1].values
c = np.where(x_consumption, 2, 0)
net_added_consumption = pd.DataFrame({net_temp.columns[0]: a,
                                        net_temp.columns[1]: b,
                                        net_temp.columns[2]: c})

a = net_temp.iloc[max_links:, 0].values
b = net_temp.iloc[max_links:, 1].values
c = np.where(x_production, 3, 0)
net_added_production = pd.DataFrame({net_temp.columns[0]: a,
                                        net_temp.columns[1]: b,
                                        net_temp.columns[2]: c})
net_ori = pd.concat([net_added_consumption, net_added_production])

####### Optimised network
x_consumption = x_optim_best[:max_links]
x_production = x_optim_best[max_links:]

a = net_temp.iloc[:max_links, 0].values
b = net_temp.iloc[:max_links, 1].values
c = np.where(x_consumption, 2, 0)
net_added_consumption = pd.DataFrame({net_temp.columns[0]: a,
                                        net_temp.columns[1]: b,
                                        net_temp.columns[2]: c})

a = net_temp.iloc[max_links:, 0].values
b = net_temp.iloc[max_links:, 1].values
c = np.where(x_production, 3, 0)
net_added_production = pd.DataFrame({net_temp.columns[0]: a,
                                        net_temp.columns[1]: b,
                                        net_temp.columns[2]: c})
net_optim = pd.concat([net_added_consumption, net_added_production])


# %%
######### Change in error with additions and deletions
f = 0.5#f_arr[0]
ec_corr_old, ct_full_old, mean_error_old, metabolome_pred_old, metabolome_measured_old = run_network_model(f, diet, cl, cellnum_init_all[0], cellnum_final_all[0], net_ori, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

ec_corr, ct_full, mean_error, metabolome_pred, metabolome_measured = run_network_model(f, diet, cl, cellnum_init_all[0], cellnum_final_all[0], net_optim, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

print('Metabolome deviation with old network is: ', mean_error_old)
print('Metabolome deviation with improved network is: ', mean_error)
print('------------------------------------------------------------------------')

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(7, 4))
ax[0].loglog(met_pred_before, met_meas_before, 'ko')
# ax[0].scatter(np.log10(metabolome_pred_old+1e-7), np.log10(metabolome_measured_old+1e-7), c='k', s=9)
ax[0].axline((-2, -2), (3, 3), c='k')
ax[0].set_title('Old network')
# ax[0].set_xlabel(r'$log_{10}\ Predicted\ metabolome$')
ax[0].set_ylabel(r'$log_{10}\ Empirical\ data$')

ax[1].loglog(met_pred_after, met_meas_after, 'ko')
# ax[1].scatter(np.log10(metabolome_pred+1e-7), np.log10(metabolome_measured+1e-7), c='k', s=9)
ax[1].axline((-2, -2), (3, 3), c='k')
ax[1].set_title('New network')
# ax[1].set_xlabel(r'$log_{10}\ Predicted\ metabolome$')
# ax[1].set_ylabel(r'$log_{10}\ Empirical\ data$')
fig.supxlabel(r'$log_{10}\ Predicted\ metabolome$')
plt.tight_layout()

if figsave_flag:
    fig.savefig(fig_path+'/no-penalties-prediction-comparison.png', dpi=300)
    plt.close(fig)
else:
    plt.show()
# plt.scatter(np.log10(metabolome_pred+1e-7), np.log10(metabolome_measured+1e-7), c='k', s=4)
    
    
# %%
df_summary = pd.concat([df_con_links, df_pro_links])

all_plots.plot_networks(net_ori, df_summary, 'original-net', fig_path, figsave_flag)
all_plots.plot_networks(net_optim, df_summary, 'optimised-net', fig_path, figsave_flag)

# %%
pickle_path = '../raw-output/'+str(k)+'-celltypes/'+net_state
try:
    os.makedirs(pickle_path)
except:
    pass

pickle_out = open(pickle_path + "optimised_network_output.pickle","wb")
#pickle.dump([net, i_selfish, i_intake, names], pickle_out)
pickle.dump([x_ori_list, x_optim_list, n_reps, MAX_ID_celltypes, MAX_ID_metabolites], pickle_out, protocol=2)
pickle_out.close()




# %%
####### Sensitivity plots
n_reps = 100
bias = ec_metabolome.iloc[:, 0].values - diet.values
net_raw = generate_random_network(all_random_networks[0], bias)

all_params = np.array([0.5, cell_line_names[0],
              cellnum_init_all[0], cellnum_final_all[0],
              net_raw, diet, in_degree_flag,
              MAX_ID_metabolites, MAX_ID_celltypes,
              0.0003, 0., 0.], dtype=object)

#### Sensitivity to kT-no penalty or reward
kT_arr = np.repeat(np.array([1e-5, 0.5*1e-4, 1e-4, 0.5*1e-3, 1e-3]), n_reps)
iter_params_list_kt = np.repeat(all_params[np.newaxis, :], len(kT_arr), 0)
iter_params_list_kt[:, -3] = kT_arr.copy()

#### Sensitivity to penalty-no reward and kT=0.0003
penalty_arr = np.repeat(np.array([1e-5, 0.5*1e-4, 1e-4, 0.5*1e-3, 1e-3]), n_reps)
iter_params_list_pn = np.repeat(all_params[np.newaxis, :], len(penalty_arr), 0)
iter_params_list_pn[:, -2] = penalty_arr.copy()

#### Sensitivity to reward-no penalty and kT=0.0003
reward_arr = np.repeat(np.array([1e-5, 0.5*1e-4, 1e-4, 0.5*1e-3, 1e-3]), n_reps)
iter_params_list_rw = np.repeat(all_params[np.newaxis, :], len(reward_arr), 0)
iter_params_list_rw[:, -1] = reward_arr.copy()

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    pool = Pool(10)
    kt_errors = pool.map(run_network_optimisation, iter_params_list_kt) #iterate over combinations
    penalty_errors = pool.map(run_network_optimisation, iter_params_list_pn) #iterate over combinations
    reward_errors = pool.map(run_network_optimisation, iter_params_list_rw) #iterate over combinations
    pool.close()
    pool.join()

# %%
kt_errors = np.array(kt_errors)
penalty_errors = np.array(penalty_errors)
reward_errors = np.array(reward_errors)

# %%
##### Byproduct fraction vs celltype number
f_arr = np.linspace(0.1, 1, 5)

all_params = np.array([0., cell_line_names[0],
              cellnum_init_all[0], cellnum_final_all[0],
              net_raw, diet, in_degree_flag,
              MAX_ID_metabolites, MAX_ID_celltypes,
              0.0003, 0., 0.], dtype=object)
iter_params_list = np.repeat(all_params[np.newaxis, :], len(f_arr), 0)
iter_params_list[:, 0] = f_arr.copy()

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    pool = Pool(10)
    f_errors = pool.map(run_network_optimisation, iter_params_list) #iterate over combinations
    pool.close()
    pool.join()

# %%
pickle_path = '../raw-output/'+str(k)+'-celltypes/'+net_state
try:
    os.makedirs(pickle_path)
except:
    pass

pickle_out = open(pickle_path + "sensitivity-kt-penalty-reward.pickle","wb")
#pickle.dump([net, i_selfish, i_intake, names], pickle_out)
pickle.dump([kt_errors, penalty_errors, reward_errors], pickle_out, protocol=2)
pickle_out.close()

pickle_out = open(pickle_path + "by-product-fraction.pickle","wb")
#pickle.dump([net, i_selfish, i_intake, names], pickle_out)
pickle.dump([f_errors], pickle_out, protocol=2)
pickle_out.close()
# %%
