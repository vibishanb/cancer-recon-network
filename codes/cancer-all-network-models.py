#!/usr/bin/env python3.12
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
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from scipy.sparse import csr_matrix
import pickle
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr, mode
import networkx as nx
from multiprocessing import Pool
import warnings
from itertools import compress, combinations

import os
import numpy.matlib
# import all_plots
from tqdm import tqdm

# %%
############ Figure size settings
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 17

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
# pickle_in = open("cancer_network.pickle","rb")


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

    return net_final, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites

def calculate_metabolome_from_net(f, ct_hyp, diet, net, prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
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
    b2m_rand = b2m * prod_rates_rand # trying out random production rate uniformly distributed in (0, 1) for each celltype-metabolite combination
    output_matrix = b2m_rand * out_mult

    ###### Final metabolome-secreted + unused
    secreted_metabolome = np.dot(output_matrix.T, intake_vector)
    i_unused = np.where(in_degree == 0)[0] # Metabolites not consumed by any celltype
    metabolome_unused = np.zeros(len(diet.values))
    metabolome_unused[i_unused] += diet.values[i_unused]
    metabolome_pred = secreted_metabolome + metabolome_unused

    return [m2b, b2m, metabolome_pred]

def calc_pred_error(ct_hyp, net, f, diet, ec_real, prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
    '''
    pred_error is a function used to compute the logarithmic error between experimentally measured
    metagenome and predicted metagenome computed from the model for a certain nutrient intake. It relies on 
    (1) diet: the nutrient intake, (2) i_intake: IDs of the nutrient intake, (3) m2b_total: a conversion matrix 
    from the nutrient intake to the total biomass, and (4) ct_hyp: hypothesised relative cell type frequenices. The 
    first three is used to compute the net gain in the intracellular metabolome predicted by the model "ic_pred" and compare it with the 
    experimentally measured net gain in intracellular metabolome "ic_real".
    '''

    m2b, b2m, ec_pred = calculate_metabolome_from_net(f, ct_hyp, diet, net, prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
    
    i_nonzero = np.where((ec_pred * ec_real) > 0, True, False)
    i_filt = np.where((m2b.sum(0) > 0), True, False)
    i_final = i_nonzero * i_filt
    diff = np.log10(ec_pred[i_final]) - np.log10(ec_real[i_final]) # / np.log10(ec_real[i_nonzero])
    pred_error = np.sqrt(np.mean(diff**2)) #np.sqrt(np.dot(pred_error, pred_error.T)) #np.sqrt(np.sum(pred_error**2))
    
    return pred_error

def run_network_model(f, diet, col_name, cellnum_init, cellnum_max, net, prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
    ######## Initial cell type frequencies and empirically measured extracellular metabolome for cell line given by col_name
    ct0 = cellnum_init*celltypefreq.values
    ec_real = ec_metabolome[col_name].values

    ct_final = np.zeros_like(ct0) # Final cell number, either fitted or taken depending on number of cell types

    ##### For max celltypes > 1, the model is converted into an optimization problem where the celltype frequencies are learned to minimize the logarithmic error between experimentally measured metabolome and predicted metabolome computed from the model for a certain up-sec network and initial cell type distribution.
    if MAX_ID_celltypes > 1:
        my_args = (net, f, diet, ec_real, prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
        bnds = ((cellnum_init, cellnum_max), ) * len(ct0)
        constraint = {'type': 'eq', 'fun': lambda ct: ct.sum() - cellnum_max}
        res = minimize(calc_pred_error, ct0, args=my_args, method='SLSQP', bounds=bnds, options={'disp': False, 'maxiter': 1000}, tol=1e-3, constraints=constraint)
        ct_final = res.x #res.x.max()/cellnum_max
        
    #### For max celltypes = 1, no model fitting, final cell number is taken directly from cell number at confluency
    else:
        ct_final = cellnum_max*celltypefreq.values
    
    ######## Compute matrices and predicted metabolome from learnt celltype abundances
    m2b, b2m, metabolome_pred_unfilt = calculate_metabolome_from_net(f, ct_final, diet, net, prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
    metabolome_measured_unfilt = ec_real.copy()

    ### Taking out zeroes from the analysis
    i_nonzero = np.where((metabolome_pred_unfilt * metabolome_measured_unfilt) > 0, True, False)
    i_filt = np.where((m2b.sum(0) > 0), True, False)
    i_final = i_nonzero * i_filt
    metabolome_measured = metabolome_measured_unfilt[i_final]
    metabolome_pred = metabolome_pred_unfilt[i_final]

    ### Correlation between predicted and expected metabolome
    ec_corr = pearsonr(np.log10(metabolome_pred), np.log10(metabolome_measured))[0]
    ### Bias in metabolome calculated for all non-trivial predictions
    bias_metabolome = np.zeros_like(metabolome_pred_unfilt)
    bias_metabolome[i_final] = np.log10(metabolome_pred) - np.log10(metabolome_measured)
    ### Mean squared error in metabolome prediction only for all non-trivial predictions
    # diff = np.log10(metabolome_pred) - np.log10(metabolome_measured)
    mean_error = np.sqrt(np.mean(bias_metabolome[i_final]**2))

    n_predicted = len(metabolome_pred)

    return [ec_corr, ct_final, mean_error, metabolome_pred_unfilt, metabolome_measured_unfilt, i_final, bias_metabolome, n_predicted]#, slope_filt, intercept_filt

####### Error function for GutCP-based algorithm
def pred_error_addingLinks(n_pred_init, residual_init, x, net_ori, f, col_name, diet, prod_rates_rand, in_degree_flag, cellnum_init, cellnum_max, pred_params):
    '''
    pred_error_addingLinks is a function used to compute the rms error between experimentally measured
    metagenome and predicted metagenome computed from the model for a certain nutrient intake. It relies on 
    (1) x: the nutrient intake, (2) i_intake: IDs of the nutrient intake, (3) m2b_total: a conversion matrix 
    from thenutrient intake to the total biomass, and (4) b_real: experimentally measured metagenome. The 
    first three is used to compute the metagenome predicted by the model "ba_pred" and compare it with the 
    experimentally measured metagenome "b_real".
    '''

    max_links = MAX_ID_celltypes * MAX_ID_metabolites # maximal number of links = number of celltypes * number of metabolites
    ec_real = ec_metabolome.loc[:, col_name].values

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

    ## RMSE calculation for all celltypes
    ec_corr, ct_full, mean_error, metabolome_pred, metabolome_measured, i_final, bias_metabolome, n_pred_new = run_network_model(f, diet, col_name, cellnum_init, cellnum_max, net, prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

    ## Collapse into a single celltype and calculate RMSE for that
    ## This is being done by taking all the consumption and production links across celltypes and assigning them to the first celltype, while keeping the other celltypes as dummy placeholders. This makes it easier to calculate the RMSE for the combined network in place without having to reassign MAX_ID_celltypes and other celltype number-specific function arguments.
    net_consumption = net.iloc[:max_links, :].groupby('metabolites').sum().clip(0, 2).reset_index()
    net_consumption.loc[:, 'celltypes'] = 0
    net_ref = net_consumption.copy()
    for i in range(1, MAX_ID_celltypes):
        net_dummy = net_ref.copy()
        net_dummy.iloc[:, -1] = 0
        net_dummy.loc[:, 'celltypes'] = i
        net_consumption = pd.concat([net_consumption, net_dummy])

    net_production = net.iloc[max_links:, :].groupby('metabolites').sum().clip(0, 3).reset_index()
    net_production.loc[:, 'celltypes'] = 0
    net_ref = net_production.copy()
    for i in range(1, MAX_ID_celltypes):
        net_dummy = net_ref.copy()
        net_dummy.iloc[:, -1] = 0
        net_dummy.loc[:, 'celltypes'] = i
        net_production = pd.concat([net_production, net_dummy])

    net_combined = pd.concat([net_consumption, net_production])
    ct_final = np.concatenate([[cellnum_max], np.zeros(MAX_ID_celltypes-1)]).flatten() # Assumed final celltype distribution reflecting a single celltype
    m2b_combined, b2m_combined, metabolome_pred_combined = calculate_metabolome_from_net(f, ct_final, diet, net_combined, prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
    i_nonzero = np.where((metabolome_pred_combined * ec_real) > 0, True, False)
    i_filt = np.where((m2b_combined.sum(0) > 0), True, False)
    i_final = i_nonzero * i_filt
    diff = np.log10(metabolome_pred_combined[i_final]) - np.log10(ec_real[i_final])
    mean_error_combined = np.sqrt(np.mean(diff**2))


    ## Balance calculation
    # b_low = 0.1
    # b_high = 1
    balance = np.zeros(MAX_ID_celltypes)
    # n_inter = 0
    # n_non_inter = 0
    for i in range(MAX_ID_celltypes):
        net_temp = net[net['celltypes']==i]
        i_consumed = np.where(net_temp.iloc[:MAX_ID_metabolites, -1]==2)[0]
        i_produced = np.where(net_temp.iloc[MAX_ID_metabolites:, -1]==3)[0]
        net_consumption = diet.values[i_consumed].sum()
        net_production = ec_real[i_produced].sum()
        balance[i] = net_production/net_consumption

        # ## Strict interconversion
        # n_tot = len(i_consumed) + len(i_produced)
        # n_inter += np.intersect1d(i_consumed, i_produced).shape[0]#np.isin(i_produced, i_consumed).sum()
        # n_non_inter += n_tot - n_inter #(~np.isin(i_produced, i_consumed)).sum()

    residual_new = np.abs(balance - 1).sum() # This residual term will be zero when the balance lies within the (0.1, 1) range for all celltypes
    
    ## Overlap calculation
    prod_links = x[max_links:].reshape(MAX_ID_celltypes, -1)
    overlap_list = np.array([prod_links[:, i].sum() for i in range(MAX_ID_metabolites)])
    num_prod_overlap = len(overlap_list[overlap_list > 1])
    
    ## Consumption overlap
    con_links = x[:max_links].reshape(MAX_ID_celltypes, -1)
    overlap_list = np.array([con_links[:, i].sum() for i in range(MAX_ID_metabolites)])
    num_con_overlap = len(overlap_list[overlap_list > 1])


    # # ## Strict interconversion
    # # n_inter = np.isin(i_produced, i_consumed).sum()
    # # n_non_inter = (~np.isin(i_produced, i_consumed)).sum()
    # inter_diff = n_inter - n_inter_old
    # non_inter_diff = n_non_inter - n_non_inter_old
    residual_diff = residual_new - residual_init

    penalty_param = pred_params['penalty']
    reward_param = pred_params['reward']
    pred_errorTotal = mean_error - np.where(n_pred_new >= n_pred_init, reward_param, -penalty_param) #+ np.where(residual_new != 0, penalty_param * np.log10(residual_new), 0) #np.where(residual_init > residual_new, penalty_param, -reward_param) #- (inter_diff*reward_param) + (non_inter_diff*penalty_param)
    # + (penalty_param * n_non_inter) - (reward_param * n_inter)
    
    return [pred_errorTotal, metabolome_pred, metabolome_measured, i_final, bias_metabolome, mean_error, mean_error_combined, n_pred_new, residual_new, num_prod_overlap, num_con_overlap]

def calculate_priors(bias_metabolome, prod_celltypes):
    # Prior probability is simply a rescaled value of the bias for each metabolite; cell type frequency not included here because there is no reference "measured" value unlike the metabolite levels
    # In a later modification of this calculation where production partitioning is hard-coded into the model, we do not want production links to be dropped or added at random. Instead, we want resampling and learning on the production side to also be within the partitioned regime. So the prior probabilities for each celltype's production are only assigned to those metabolites that are in that celltype's production range.
    prior_temp = np.exp(np.abs(bias_metabolome))/np.exp(np.abs(bias_metabolome)).sum(0)

    ## Consumption priors are random and same for all celltypes
    consumption_prior, production_prior = np.zeros(MAX_ID_metabolites), np.zeros(MAX_ID_metabolites)
    i_con_prior = np.where(bias_metabolome > 0)[0]
    consumption_prior[i_con_prior] = prior_temp[i_con_prior]
    consumption_prior = np.concatenate([consumption_prior]*MAX_ID_celltypes)

    # ## Production priors without partitioning
    # i_pro_prior = np.where(bias_metabolome < 0)[0]
    # production_prior[i_pro_prior] = prior_temp[i_pro_prior]
    # production_prior = np.repeat(production_prior, MAX_ID_celltypes)

    ## Production priors with partitioning
    prod_prior_temp = []
    for i in range(MAX_ID_celltypes):
        prod_prior_temp.append(np.where(prod_celltypes == i+1, prior_temp, 0))
    production_prior = np.array(prod_prior_temp).ravel()

    prior_prob = np.concatenate([consumption_prior, production_prior]) # Appending the same array twice, first for consumption links and then for production links i.e., prior probability for a given metabolite depends on the error in prediction but is the same for consumption and production links
    max_links = MAX_ID_celltypes * MAX_ID_metabolites 
    prior_prob += 1/max_links # All links have some constant probability to be chosen at random, independent of the bias in the metabolome prediction
    prior_prob /= prior_prob.sum(0) # Normalise to [0, 1]
    return prior_prob

def run_network_optimisation(all_params):
#(f, cl, cellnum_init, cellnum_final, net_raw, diet, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
    f = all_params[0]
    cl = all_params[1]
    cellnum_init = all_params[2]
    cellnum_final = all_params[3]
    net_ori = all_params[4]
    diet = all_params[5]
    in_degree_flag = all_params[6]
    MAX_ID_metabolites = all_params[7]
    MAX_ID_celltypes = all_params[8]
    
    error_list = []
    prior_list = []
    log_bias_list = []
    log_bias_combined_list = []
    metabolome_pred_list = []
    metabolome_measured_list = []
    valid_index_list = []
    n_pred_list = []
    residual_list = []
    x_list = []
    con_overlap_list, prod_overlap_list = [], []
    
    ct0 = cellnum_init*celltypefreq.to_numpy()

    # Save original network features and prediction errors
    # net_ori, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(net_raw)
    prod_rates_rand = all_params[-1]
    m2b, b2m, met_pred = calculate_metabolome_from_net(f, ct0, diet, net_ori, prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
    ####### Keep a record of the original network
    m2b_ori = (m2b!=0).astype(int).copy()
    b2m_ori = (b2m!=0).astype(int).copy()
    x_ori = np.concatenate([m2b_ori.flatten(), b2m_ori.flatten()]) # This is the adjacency matrix
    
    ##### How many metabolites predicted non-trivially in the initial network
    metabolome_measured = ec_metabolome.loc[:, cl].values
    i_nonzero = np.where(met_pred * metabolome_measured, True, False)
    i_filt = np.where((m2b.sum(0) > 0), True, False)
    i_final = i_nonzero * i_filt
    n_pred_init = i_final.sum()

    # Bias in the metabolome is calculated as the difference between predicted and measured metabolite levels
    max_links = m2b.shape[0]*m2b.shape[1]
    pred_params = {'penalty': all_params[10], 'reward': all_params[11]}

    ## Balance calculation
    balance = np.zeros(MAX_ID_celltypes)
    for i in range(MAX_ID_celltypes):
        net_temp = net_ori[net_ori['celltypes']==i]
        i_consumed = np.where(net_temp.iloc[:MAX_ID_metabolites, -1]==2)[0]
        i_produced = np.where(net_temp.iloc[MAX_ID_metabolites:, -1]==3)[0]
        net_consumption = diet.values[i_consumed].sum()
        net_production = metabolome_measured[i_produced].sum()
        balance[i] = net_production/net_consumption
    residual_init = np.abs(balance - 1).sum() # This residual term will be zero when the balance becomes 1 for all celltypes

    error_before, metabolome_pred, metabolome_measured, i_final, bias_metabolome, log_bias_init, log_bias_combined_init, n_pred, residual_init, num_prod_overlap, num_con_overlap = pred_error_addingLinks(n_pred_init, residual_init, x_ori, net_ori, f, cl, diet, prod_rates_rand, in_degree_flag, cellnum_init, cellnum_final, pred_params)
    metabolome_pred_list.append(metabolome_pred)
    metabolome_measured_list.append(metabolome_measured)
    valid_index_list.append(i_final)
    log_bias_list.append(log_bias_init)
    log_bias_combined_list.append(log_bias_combined_init)
    n_pred_list.append(n_pred)
    residual_list.append(residual_init)
    x_list.append(x_ori)
    bias_metabolome_ori = bias_metabolome.copy()
    prod_overlap_list.append(num_prod_overlap)
    con_overlap_list.append(num_con_overlap)

    ## Partitioning of the metabolome based on number of celltypes assumed; The array prod_celltype is a vector that codes which celltype is assumed to produce each metabolite under partition.
    quantiles = np.quantile(metabolome_measured, np.linspace(0, 1, MAX_ID_celltypes+1)[1:-1])
    prod_celltypes = np.zeros_like(metabolome_measured)
    for i in range(MAX_ID_celltypes-1):
        prod_celltypes = np.where(metabolome_measured <= quantiles[i], prod_celltypes, prod_celltypes+1)
    prod_celltypes = np.int64(prod_celltypes + 1)

    prior_prob = calculate_priors(bias_metabolome_ori, prod_celltypes)
    # print('The original error is', error_before)
    error_list.append(error_before)
    x = x_ori.copy()

    # Network optimisation begins here
    kT = all_params[9]
    Twindow = 500
    error_window = []
    for i in range(20000):
        linknum = np.random.randint(2, n_ct)
        error_window.append(error_before)
        x_previous = x.copy()
        if np.random.uniform(0,1,1)[0] <= 0.5: # Each step chooses randomly between adding a new link or removing an existing link
            i_x = np.random.choice(np.where(x==0)[0], linknum, p=prior_prob[x==0]/prior_prob[x==0].sum())[0] # New link is chosen based on the prior probability-smaller bias in prediction leads to smaller prior prob
            x[[i_x]] = np.ones_like(i_x)
        elif (n_pred_init >= 4) * ((x==1).sum() > 0): # Only propose deletion if at least three metabolites can be non-trivially predicted after deletion
            i_x = np.random.choice(np.where(x==1)[0], linknum, p=prior_prob[x==1]/prior_prob[x==1].sum())[0] # New link is chosen based on the prior probability-smaller bias in prediction leads to smaller prior prob
            x[[i_x]] = np.zeros_like(i_x)
        else:
            continue

        error_after, metabolome_pred, metabolome_measured, i_final, bias_metabolome, log_bias, log_bias_combined, n_pred, residual_after, num_prod_overlap, num_con_overlap = pred_error_addingLinks(n_pred_init, residual_init, x, net_ori, f, cl, diet, prod_rates_rand, in_degree_flag, cellnum_init, cellnum_final, pred_params) # Calculate prediction error with the modified network
        prior_prob = calculate_priors(bias_metabolome, prod_celltypes)

        if (n_pred >= 3) * ((error_after - error_before) <= -0.025):#* (np.random.uniform(0,1,1)[0] < np.exp((error_before-error_after)/kT)): # If at least three metabolites are non-trivially predicted and the reduction in error is large enough, the proposed link addition/removal is accepted
            error_before = error_after
            error_list.append(error_before)
            prior_list.append(prior_prob[[i_x]])
            log_bias_list.append(log_bias)
            log_bias_combined_list.append(log_bias_combined)
            metabolome_pred_list.append(metabolome_pred)
            metabolome_measured_list.append(metabolome_measured)
            valid_index_list.append(i_final)
            n_pred_list.append(n_pred)
            residual_list.append(residual_after)
            prod_overlap_list.append(num_prod_overlap)
            con_overlap_list.append(num_con_overlap)
            x_list.append(x.copy())
            n_pred_init = n_pred
            residual_init = residual_after.copy()

        else:  ## not accepted   
            x[[i_x]] = x_previous[[i_x]] # Maintain the previous state
            n_pred_init = n_pred
            residual_init = residual_after.copy()

        if (i > Twindow) and (error_window[-1] - error_window[-Twindow]) > -0.067:#-(np.sqrt(Twindow)*kT)):
            break
    
    return [kT, pred_params['penalty'], pred_params['reward'], f, x_ori, x, x_list, error_list, n_pred_list, residual_list, prod_overlap_list, con_overlap_list, metabolome_pred_list, metabolome_measured_list, valid_index_list, log_bias_list, log_bias_combined_list, error_window]#, consumption_added, production_added, consumption_deleted, production_deleted]

def run_null_optimisation(all_params):
#(f, cl, cellnum_init, cellnum_final, net_raw, diet, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
    f = all_params[0]
    cl = all_params[1]
    cellnum_init = all_params[2]
    cellnum_final = all_params[3]
    net_ori = all_params[4]
    diet = all_params[5]
    in_degree_flag = all_params[6]
    MAX_ID_metabolites = all_params[7]
    MAX_ID_celltypes = all_params[8]
    
    error_list = []
    # prior_list = []
    log_bias_list = []
    # log_bias_combined_list = []
    # metabolome_pred_list = []
    # metabolome_measured_list = []
    # valid_index_list = []
    n_pred_list = []
    # residual_list = []
    x_list = []
    con_overlap_list, prod_overlap_list = [], []
    
    ct0 = cellnum_init*celltypefreq.to_numpy()

    # Save original network features and prediction errors
    # net_ori, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(net_raw)
    prod_rates_rand = all_params[-2]
    m2b, b2m, met_pred = calculate_metabolome_from_net(f, ct0, diet, net_ori, prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
    ####### Keep a record of the original network
    m2b_ori = (m2b!=0).astype(int).copy()
    b2m_ori = (b2m!=0).astype(int).copy()
    x_ori = np.concatenate([m2b_ori.flatten(), b2m_ori.flatten()]) # This is the adjacency matrix
    
    ##### How many metabolites predicted non-trivially in the initial network
    metabolome_measured = ec_metabolome.loc[:, cl].values
    i_nonzero = np.where(met_pred * metabolome_measured, True, False)
    i_filt = np.where((m2b.sum(0) > 0), True, False)
    i_final = i_nonzero * i_filt
    n_pred_init = i_final.sum()

    # Bias in the metabolome is calculated as the difference between predicted and measured metabolite levels
    max_links = m2b.shape[0]*m2b.shape[1]
    pred_params = {'penalty': all_params[10], 'reward': all_params[11]}

    ## Balance calculation
    balance = np.zeros(MAX_ID_celltypes)
    for i in range(MAX_ID_celltypes):
        net_temp = net_ori[net_ori['celltypes']==i]
        i_consumed = np.where(net_temp.iloc[:MAX_ID_metabolites, -1]==2)[0]
        i_produced = np.where(net_temp.iloc[MAX_ID_metabolites:, -1]==3)[0]
        net_consumption = diet.values[i_consumed].sum()
        net_production = metabolome_measured[i_produced].sum()
        balance[i] = net_production/net_consumption
    residual_init = np.abs(balance - 1).sum() # This residual term will be zero when the balance becomes 1 for all celltypes

    error_before, metabolome_pred, metabolome_measured, i_final, bias_metabolome, log_bias_init, log_bias_combined_init, n_pred, residual_init, num_prod_overlap, num_con_overlap = pred_error_addingLinks(n_pred_init, residual_init, x_ori, net_ori, f, cl, diet, prod_rates_rand, in_degree_flag, cellnum_init, cellnum_final, pred_params)
    # metabolome_pred_list.append(metabolome_pred)
    # metabolome_measured_list.append(metabolome_measured)
    # valid_index_list.append(i_final)
    log_bias_list.append(log_bias_init)
    # log_bias_combined_list.append(log_bias_combined_init)
    n_pred_list.append(n_pred)
    # residual_list.append(residual_init)
    x_list.append(x_ori)
    bias_metabolome_ori = bias_metabolome.copy()
    # prod_overlap_list.append(num_prod_overlap)
    # con_overlap_list.append(num_con_overlap)

    ## Partitioning of the metabolome based on number of celltypes assumed; The array prod_celltype is a vector that codes which celltype is assumed to produce each metabolite under partition.
    quantiles = np.quantile(metabolome_measured, np.linspace(0, 1, MAX_ID_celltypes+1)[1:-1])
    prod_celltypes = np.zeros_like(metabolome_measured)
    for i in range(MAX_ID_celltypes-1):
        prod_celltypes = np.where(metabolome_measured <= quantiles[i], prod_celltypes, prod_celltypes+1)
    prod_celltypes = np.int64(prod_celltypes + 1)

    prior_prob = calculate_priors(bias_metabolome_ori, prod_celltypes)
    # print('The original error is', error_before)
    error_list.append(error_before)
    x = x_ori.copy()

    # Network optimisation begins here
    kT = all_params[9]
    Twindow = all_params[-1]#500
    error_window = []
    for i in range(20000):
        linknum = np.random.randint(2, n_ct)
        error_window.append(error_before)
        x_previous = x.copy()
        if np.random.uniform(0,1,1)[0] <= 0.5: # Each step chooses randomly between adding a new link or removing an existing link
            i_x = np.random.choice(np.where(x==0)[0], linknum, p=prior_prob[x==0]/prior_prob[x==0].sum())[0] # New link is chosen based on the prior probability-smaller bias in prediction leads to smaller prior prob
            x[[i_x]] = np.ones_like(i_x)
        elif (n_pred_init >= 4) * ((x==1).sum() > 0): # Only propose deletion if at least three metabolites can be non-trivially predicted after deletion
            i_x = np.random.choice(np.where(x==1)[0], linknum, p=prior_prob[x==1]/prior_prob[x==1].sum())[0] # New link is chosen based on the prior probability-smaller bias in prediction leads to smaller prior prob
            x[[i_x]] = np.zeros_like(i_x)
        else:
            continue

        error_after, metabolome_pred, metabolome_measured, i_final, bias_metabolome, log_bias, log_bias_combined, n_pred, residual_after, num_prod_overlap, num_con_overlap = pred_error_addingLinks(n_pred_init, residual_init, x, net_ori, f, cl, diet, prod_rates_rand, in_degree_flag, cellnum_init, cellnum_final, pred_params) # Calculate prediction error with the modified network
        # prior_prob = calculate_priors(bias_metabolome, prod_celltypes)

        if (n_pred >= 3) * (np.random.random() > 0.5):
            error_before = error_after
            error_list.append(error_before)
            # prior_list.append(prior_prob[[i_x]])
            log_bias_list.append(log_bias)
            # log_bias_combined_list.append(log_bias_combined)
            # metabolome_pred_list.append(metabolome_pred)
            # metabolome_measured_list.append(metabolome_measured)
            # valid_index_list.append(i_final)
            n_pred_list.append(n_pred)
            # residual_list.append(residual_after)
            # prod_overlap_list.append(num_prod_overlap)
            # con_overlap_list.append(num_con_overlap)
            x_list.append(x.copy())
            n_pred_init = n_pred
            residual_init = residual_after.copy()

        else:  ## not accepted   
            x[[i_x]] = x_previous[[i_x]] # Maintain the previous state
            n_pred_init = n_pred
            residual_init = residual_after.copy()

        if (i > Twindow): #and (error_window[-1] - error_window[-Twindow]) > -0.067:#-(np.sqrt(Twindow)*kT)):
            break
    
    return [x_list, error_list, n_pred_list, log_bias_list]#, consumption_added, production_added, consumption_deleted, production_deleted]


def net_from_x(x, MAX_ID_metabolites, MAX_ID_celltypes):

    max_links = MAX_ID_celltypes * MAX_ID_metabolites # maximal number of links = number of celltypes * number of metabolites

    ######## Convert x to net structure (convert the adjacency matrix into the edge list):
    ### consumption links:
    x_consumption = x[:max_links]
    x_production = x[max_links:]

    a = np.repeat(np.arange(MAX_ID_metabolites)[np.newaxis, :], MAX_ID_celltypes, axis=0).ravel() #net_ori.iloc[:max_links, 0].values
    b = np.repeat(np.arange(MAX_ID_celltypes), MAX_ID_metabolites) #net_ori.iloc[:max_links, 1].values
    c = np.where(x_consumption, 2, 0)
    net_added_consumption = pd.DataFrame({'metabolites': a,
                                          'celltypes': b,
                                          'edgeType': c})
    
    # a = net_ori.iloc[max_links:, 0].values
    # b = net_ori.iloc[max_links:, 1].values
    c = np.where(x_production, 3, 0)
    net_added_production = pd.DataFrame({'metabolites': a,
                                          'celltypes': b,
                                          'edgeType': c})

    net = pd.concat([net_added_consumption, net_added_production])

    return net

def calculate_overlap_stats(x, n_ct, max_links, met_ID):

    x_consumption = x[:max_links]
    x_production = x[max_links:]

    ### Consumption overlap
    con_index_overlap, con_mets_overlap = [[]], []
    for i, arr in enumerate(x_consumption.reshape(n_ct, -1).T):
        indices = np.where(arr == 1)[0]
        if len(indices) > 1:
            con_index_overlap.append(indices+1)
            con_mets_overlap.append(met_ID[i])
    
    ### Production overlap
    pro_index_overlap, pro_mets_overlap = [[]], []
    for i, arr in enumerate(x_production.reshape(n_ct, -1).T):
        indices = np.where(arr == 1)[0]
        if len(indices) > 1:
            pro_index_overlap.append(indices+1)
            pro_mets_overlap.append(met_ID[i])
    
    clist = ["+".join(map(str, ind.tolist())) for ind in con_index_overlap[1:]]
    plist = ["+".join(map(str, ind.tolist())) for ind in pro_index_overlap[1:]]

    return [clist, con_mets_overlap, plist, pro_mets_overlap]



# %%
####### Adding all possible single overlap links to a network with no overlap
figsave_flag = 0
for n_ct in range(5, 6):
    home_dir = os.getcwd() #+ '/codes'
    celltype_ID, celltypefreq, ec_metabolome_ID, ec_metabolome, met_baseline, core_mean = pd.read_pickle(home_dir + '/' + str(n_ct) + '-cells-data.pickle')
    cell_line_names = ec_metabolome.columns.to_numpy()

    slope_df, sublinear_cell_lines = pd.read_pickle(home_dir + '/cancer_power_law_stats.pickle')

    ### Filtering those networks for cell lines with power law slopes
    i_sublinear = np.where(np.isin(ec_metabolome.columns.values, sublinear_cell_lines))[0]
    ec_metabolome = ec_metabolome.iloc[:, i_sublinear]

    ######## Diet as the average of all the Baseline values
    diet = met_baseline.mean(axis=1)

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

    cellnum_init_all = cellnum_init_all[i_sublinear]
    cellnum_final_all = cellnum_final_all[i_sublinear]

    cell_line_names = ec_metabolome.columns.to_numpy()
    for i_cell_line in range(1):#range(len(sublinear_cell_lines[0])):
        cl = cell_line_names[i_cell_line]
        f = 0.5
        ec_real = ec_metabolome.loc[:, cl].values

        pickle_path = '../raw-output/'+str(n_ct)+'-celltypes/no-learn-balanced-net/'+cl
        [balance_arr_1ct, balance_arr_nct, balance_flag_list, balanced_networks_list,
                            rmse_arr_1ct, rmse_arr_nct,
                            ec_pred_arr_1ct, ec_pred_arr_nct,
                            index_arr_1ct, index_arr_nct,
                            prod_rates_1ct, prod_rates_nct,
                            production_ct_arr_1ct, production_ct_arr_nct] = pd.read_pickle(pickle_path + '/balanced-networks.pickle')
        all_networks = balanced_networks_list.copy()

        i_nonzero_celltypes = all_networks[0]['celltypes'].unique()
        i_nonzero_celltypes = np.sort(i_nonzero_celltypes)
        i_nonzero_celltypes = celltype_ID.values.copy()
        i_nonzero_metabolites = all_networks[0]['metabolites'].unique()

        MAX_ID_celltypes = len(i_nonzero_celltypes)  # MAX_ID_celltypes is the maximum of ID labels for celltypes.
        MAX_ID_metabolites = len(i_nonzero_metabolites)  # MAX_ID_metabolites is the maximum of ID labels for metabolites.

        cellnum_init = cellnum_init_all[i_cell_line]
        cellnum_final = cellnum_final_all[i_cell_line]
        
        max_links = MAX_ID_celltypes * MAX_ID_metabolites
        met_ID = diet.index.to_numpy()
        in_degree_flag = False

        source = np.arange(n_ct)
        ov_len = np.arange(2, n_ct+1)
        ov_mean_error_pooled, ov_fraction_pooled, network_index_pooled, pro_con_pooled = [], [], [], []
        ov_links_pooled, ov_mets_pooled, best_worst_pooled, linktype_pooled, some_error = [], [], [], [], []
        for i_net in tqdm(np.where(balance_flag_list==True)[0], desc='Replicate: '):
            ov_networks = []
            pro_con_list = []
            # ref_net = all_networks[0].iloc[:max_links, :]
            net_ori = all_networks[i_net]
            prod_rates_rand = prod_rates_nct[i_net]
            net_production = net_ori.iloc[max_links:, :]
            net_consumption = net_ori.iloc[:max_links, :]

            for s in source:
                all_targets = [[]]
                for l in ov_len:
                    fodder = np.delete(np.arange(n_ct), s)
                    ov_targets = np.array(list(combinations(fodder, l-1)))
                    for ov in ov_targets:
                        all_targets.append(ov)
                    # all_targets.append(ov_targets)
                all_targets = np.array(all_targets[1:], dtype='object')

                ref_net = net_consumption.copy()
                consumed_mets = ref_net.iloc[np.where((ref_net['celltypes']==s)*(ref_net['edgeType']==2))[0], 0].values
                consumed_mets = consumed_mets[ec_real[consumed_mets]!=0]
                for met in consumed_mets:
                    for target in all_targets:
                        current_links = ref_net.iloc[:, -1].values
                        net_test = ref_net.copy()
                        net_test.loc[:, 'edgeType'] = np.where(np.isin(net_test['celltypes'], target)*(net_test['metabolites']==met), 2, current_links)
                        net_final = pd.concat([net_test, net_production])
                        ov_networks.append(net_final.copy())
                        pro_con_list.append('Consumption')
                
                ref_net = net_production.copy()
                produced_mets = ref_net.iloc[np.where((ref_net['celltypes']==s)*(ref_net['edgeType']==3))[0], 0].values
                produced_mets = produced_mets[ec_real[produced_mets]!=0]
                for met in produced_mets:
                    for target in all_targets:
                        current_links = ref_net.iloc[:, -1].values
                        net_test = ref_net.copy()
                        net_test.loc[:, 'edgeType'] = np.where(np.isin(net_test['celltypes'], target)*(net_test['metabolites']==met), 3, current_links)
                        net_final = pd.concat([net_consumption, net_test])
                        ov_networks.append(net_final.copy())
                        pro_con_list.append('Production')

            # #### This was a brief trial to check if the single-overlap networks assembled above actually have the overlaps we wanted using the independently-coded calculate_overlap_stats function--this validation has been done and the code is working as expected as of Apr 29, 2026

            # net_test = ov_networks[-1]
            # ct0 = cellnum_init * celltypefreq.values
            # m2b, b2m, met_pred = calculate_metabolome_from_net(f, ct0, diet, net_test, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
                
            # m2b = (m2b!=0).astype(int).copy()
            # b2m = (b2m!=0).astype(int).copy()
            # x = np.concatenate([m2b.flatten(), b2m.flatten()]) # This is the adjacency matrix
                
            # max_links = n_ct*MAX_ID_metabolites
            # met_ID = diet.index.to_numpy()
            # calculate_overlap_stats(x, n_ct, max_links, met_ID)

            ov_mean_error = []
            for ltype in ['Consumption', 'Production']:
                # ov_fraction = []
                # network_index = []
                i_networks = np.where(np.array(pro_con_list) == ltype)[0]#list(compress(ov_networks, np.where(np.array(pro_con_list) == ltype, True, False)))
                for fr in [1.]:#np.linspace(0.1, 1, 5):
                    size = int(fr * len(i_networks))
                    chosen_indices = np.random.choice(i_networks, size=size)
                    for k in chosen_indices:
                        ec, ct, ov_err, metabolome_pred, metabolome_measured, i_final, bias_metabolome, n_predicted = run_network_model(f, diet, cl, cellnum_init, cellnum_final, ov_networks[k], prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

                        ov_mean_error.append(ov_err)
                        ov_fraction_pooled.append(fr)
                        network_index_pooled.append(k)
                        pro_con_pooled.append(ltype)

                    ec, ct, mean_error, metabolome_pred, metabolome_measured, i_final, bias_metabolome, n_predicted = run_network_model(f, diet, cl, cellnum_init, cellnum_final, net_ori, prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
                    ov_mean_error_pooled = np.concatenate([ov_mean_error_pooled, np.array(ov_mean_error - mean_error)])

                    if fr == 1.0:
                        ov_mean_error = np.array(ov_mean_error - mean_error)
                        i_best = chosen_indices[np.where(ov_mean_error == ov_mean_error.min())[0][-1]]
                        # i_worst = chosen_indices[np.where(ov_mean_error == ov_mean_error.max())[0][-1]]
                    
                        net_test = ov_networks[i_best]
                        ct0 = cellnum_init * celltypefreq.values
                        m2b, b2m, met_pred = calculate_metabolome_from_net(f, ct0, diet, net_test, prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
                            
                        m2b = (m2b!=0).astype(int).copy()
                        b2m = (b2m!=0).astype(int).copy()
                        x = np.concatenate([m2b.flatten(), b2m.flatten()]) # This is the adjacency matrix
                        if ltype == 'Consumption':
                            l, m = calculate_overlap_stats(x, n_ct, max_links, met_ID)[:2]
                        else:
                            l, m = calculate_overlap_stats(x, n_ct, max_links, met_ID)[2:]
                        ov_links_pooled.append(l[0])
                        ov_mets_pooled.append(m[0])
                        best_worst_pooled.append('Best')
                        linktype_pooled.append(ltype)
                        some_error.append(ov_mean_error.min())

                        # net_test = ov_networks[i_worst]
                        # ct0 = cellnum_init * celltypefreq.values
                        # m2b, b2m, met_pred = calculate_metabolome_from_net(f, ct0, diet, net_test, prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
                            
                        # m2b = (m2b!=0).astype(int).copy()
                        # b2m = (b2m!=0).astype(int).copy()
                        # x = np.concatenate([m2b.flatten(), b2m.flatten()]) # This is the adjacency matrix
                        # if ltype == 'Consumption':
                        #     l, m = calculate_overlap_stats(x, n_ct, max_links, met_ID)[:2]
                        # else:
                        #     l, m = calculate_overlap_stats(x, n_ct, max_links, met_ID)[2:]
                        # ov_links_pooled.append(l[0])
                        # ov_mets_pooled.append(m[0])
                        # best_worst_pooled.append('Worst')
                        # linktype_pooled.append(ltype)
                        # some_error.append(ov_mean_error.max())

                    ov_mean_error = []

        fig_path = '../figures/'+str(n_ct)+'-celltypes/no-learn-balanced-net/'+cl
        ov_error_df = pd.DataFrame({'Fraction': ov_fraction_pooled,
                                'Ltype': pro_con_pooled,
                                'Index': network_index_pooled,
                                'DelRMSE': ov_mean_error_pooled})
        ### Pooled statistics-only uncomment when running all replicates from balanced_networks_list for a particular cell line
        with sns.axes_style('ticks'):
            con_ov_df = ov_error_df[(ov_error_df['Fraction']==1)*(ov_error_df['Ltype']=='Consumption')]
            delrmse_mean = con_ov_df.loc[:, 'DelRMSE'].mean()
            
            # Define the linear threshold (values inside this range are treated linearly)
            data = con_ov_df.loc[:, 'DelRMSE'].values
            linthresh = 0.2          # Size of the linear zone around zero
            num_log_bins = 6          # Number of bins per log side
            max_val = np.abs(data).max()
            # linthresh = 0.002 #mode(data)[0]
            # # Create logarithmic bins for the positive and negative regions
            # data = con_ov_df.loc[:, 'DelRMSE'].values
            # Positive log bins
            pos_bins = np.geomspace(linthresh, max_val, num=7)
            # pos_bins = np.linspace(np.log10(0.2), np.log10(max(data)), num=15)
            # Negative log bins (mirrored)
            neg_bins = -np.geomspace(linthresh, max_val, num=1)
            # neg_bins = -np.linspace(np.log10(0.02), np.log10(-min(data)), num=5)[::-1]
            # Linear region bins around zero
            center_bins = np.linspace(-linthresh, linthresh, num=5)
            # lin_bins = np.logspace(0.02, 0.2, num=5)

            # 4. Combine all segments into a single unified bin array
            symlog_bins = np.unique(np.concatenate([neg_bins, center_bins, pos_bins]))

            g = sns.histplot(data=con_ov_df, x='DelRMSE',
                                    hue='Ltype', palette={'Consumption': 'tab:green', 'Production': 'tab:blue'},
                                    stat='proportion', element='step',
                                    alpha=0.6, fill=True, bins=symlog_bins)
            g.set_xlabel(r'$\Delta$RMSE')
            g.axvline(x=0, linestyle='--', linewidth=1.5, c='tab:red')
            g.axvspan(xmin=-delrmse_mean, xmax=delrmse_mean, alpha=0.3, color='tab:red')
            sns.despine(offset=5, trim=0)
            g.get_legend().remove()
            g.set_title('Consumption overlap improvement', pad=4)
            g.figure.tight_layout()
            # g.set_xscale('symlog')
            # g.set_xscale('log')
            # g.set_xlabels(r'Link type')
            # g.set_titles(row_template="{row_name} overlap")
            # g.tight_layout()
        try:
            os.makedirs(fig_path)
        except:
            pass
        g.figure.savefig(fig_path+'/delta-rmse-consumption-hist-plot', dpi=300)
        plt.close(g.figure)

        # with sns.axes_style('whitegrid'):
        #     g = sns.catplot(data=ov_error_df[ov_error_df['Fraction']==1], x='Ltype', y='DelRMSE',
        #                 hue='Ltype', palette={'Consumption': 'tab:green', 'Production': 'tab:blue'},
        #                 kind='violin', sharey=False)
        #     g.set_ylabels(r'$\Delta$ RMSE')
        #     g.set_xlabels(r'Link type')
        #     g.set_titles(row_template="{row_name} overlap")
        #     g.tight_layout()
        # if figsave_flag:
        #     try:
        #         os.makedirs(fig_path)
        #     except:
        #         pass
        #     g.savefig(fig_path+'/delta-rmse-production-consumption-violin-plot', dpi=300)
        #     plt.close(g.figure)

        ov_links_df = pd.DataFrame({'Link': ov_links_pooled,
                                    'Mets': ov_mets_pooled,
                                    'NetType': best_worst_pooled,
                                    'LinkType': linktype_pooled})
        mets_order = np.array(diet.sort_values(inplace=False).index) # IDs of overlapping metabolites sorted by their abundances in the fresh medium
        mets_ranks = np.arange(len(mets_order)) # Assigning ranks to the sorted metabolites
        ov_links_df.loc[:, 'MetRanks'] = np.array([mets_ranks[np.where(i == mets_order)[0]][0] for i in ov_links_df['Mets'].values]) # Mapping metabolite IDs in the dataframe to ranks based on fresh medium abundaces

        unique_indices = np.unique(ov_links_df['Link'].values) # All unique overlaps
        index_len = np.array([len(i) for i in unique_indices]) # How many celltypes in each overlap
        unique_indices_sorted = [[]]
        for i in range(3, n_ct + n_ct - 1 + 1): # Range limits based on the string length for n_ct celltype overlap e.g., '1+2' is a string of length 3, '1+2+3' is of length 5, so this goes as n + (n-1), where n is the number of overlapping celltypes. The extra plus one is to account for the range function not including the last number
            if len(unique_indices[index_len == i]) > 0: # If there is an overlap of this length
                unique_indices_sorted.append(unique_indices[index_len == i]) # Retrieving all overlaps of a given length this way sorts the unique elements of that particular length
        unique_indices_sorted = np.concatenate(unique_indices_sorted[1:])

        ov_links_df['Link'] = pd.Categorical(ov_links_df['Link'], categories=unique_indices_sorted) # This sets the column to categorical with the levels in the order above, giving 2-celltype overlaps first, then three and so on, and sorted by celltype number within each overlap length

        h = sns.displot(data=ov_links_df, x='MetRanks', kind='hist',
                        hue='LinkType', palette={'Consumption': 'tab:green',
                                                                'Production': 'tab:blue'},
                        row='NetType',
                        multiple='stack', stat='probability', common_norm=False, 
                        discrete=True, height=2, aspect=4)
        h.tick_params(axis='x', labelrotation=60)
        h.set_xlabels('Abundance rank')
        if figsave_flag:
            h.savefig(fig_path+'/overlaps-met-abundance-rank-distribution', dpi=300)
            plt.close(h.figure)

        s = sns.displot(data=ov_links_df, x='Link', kind='hist',
                        hue='LinkType', palette={'Consumption': 'tab:green',
                                                                'Production': 'tab:blue'},
                        row='NetType',
                        multiple='stack', stat='probability', common_norm=False, 
                        discrete=True, shrink=0.9, height=2, aspect=3)
        s.tick_params(axis='x', labelrotation=60)
        s.set_titles('')
        s.set_xlabels('Overlapping link')
        if figsave_flag:
            s.savefig(fig_path+'/overlapping-links-distribution', dpi=300)
            plt.close(s.figure)

        # ##### Check out the best performing consumption and production overlaps for one replicate of balanced network-uncomment this code block only when running the above overlaps script for one replicate from balanced_networks_list; do this by adding an array index slice to the np.where statement in the for loop in line 646. For example, np.where(balance_flag_list == True)[0][1:2] would select the second replicate with balance from the original list.
        # ### Original network prediction-no overlaps
        # # i_net = 65
        # net_test = all_networks[i_net]
        # ct0 = cellnum_init * celltypefreq.values
        # m2b, b2m, met_pred_ori = calculate_metabolome_from_net(f, ct0, diet, net_test, prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
        # m2b = (m2b!=0).astype(int).copy()
        # b2m = (b2m!=0).astype(int).copy()
        # x_ori = np.concatenate([m2b.flatten(), b2m.flatten()]) # This is the adjacency matrix

        # ec_ori, ct_ori, mean_error_ori, met_pred_ori, met_meas_ori, i_final_ori, bias_metabolome_ori, n_predicted_ori = run_network_model(f, diet, cl, cellnum_init_all[0], cellnum_final_all[0], all_networks[i_net], prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

        # ### Picking networks with the best single consumption and production overlaps
        # con_df = ov_error_df[(ov_error_df['Fraction']==1)*(ov_error_df['Ltype']=='Consumption')]
        # pro_df = ov_error_df[(ov_error_df['Fraction']==1)*(ov_error_df['Ltype']=='Production')]

        # i_con_best = con_df.iloc[np.where(con_df['DelRMSE'] == con_df['DelRMSE'].min())[0], 2].values[-1] # Best single consumption overlap
        # # i_con_worst = con_df.iloc[np.where(con_df['DelRMSE'] == con_df['DelRMSE'].max())[0], 2].values[-1] # Worst single consumption overlap
        # i_pro_best = pro_df.iloc[np.where(pro_df['DelRMSE'] == pro_df['DelRMSE'].min())[0], 2].values[0] # Best single production overlap
        # # i_pro_worst = pro_df.iloc[np.where(pro_df['DelRMSE'] == pro_df['DelRMSE'].max())[0], 2].values[-1] # Best single production overlap

        # ### Best single consumption overlap
        # for k, i in enumerate([i_con_best, i_pro_best]): #, i_con_worst, i_pro_best, i_pro_worst]):
        #     net_ov = ov_networks[i]
        #     ct0 = cellnum_init * celltypefreq.values
        #     m2b, b2m, met_pred_ov = calculate_metabolome_from_net(f, ct0, diet, net_ov, prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

        #     m2b = (m2b!=0).astype(int).copy()
        #     b2m = (b2m!=0).astype(int).copy()
        #     x_ov = np.concatenate([m2b.flatten(), b2m.flatten()]) # This is the adjacency matrix

        #     ec_ov, ct_ov, mean_error_ov, met_pred_ov, met_meas_ov, i_final_ov, bias_metabolome_ov, n_predicted_ov = run_network_model(f, diet, cl, cellnum_init_all[0], cellnum_final_all[0], net_ov, prod_rates_rand, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
        #     i_before = i_final_ori.astype(bool)
        #     i_after = i_final_ov.astype(bool)

        #     p_arr_ori = np.array([i*j for i, j in zip(x_ori[max_links:].reshape(n_ct, -1), np.arange(1, n_ct+1).tolist())]).sum(0)
        #     p_arr_ov = np.array([i*j for i, j in zip(x_ov[max_links:].reshape(n_ct, -1), np.arange(1, n_ct+1).tolist())]).sum(0)
        #     c_before = np.where(p_arr_ori == 0, 'tab:gray', np.where(p_arr_ori == 1, 'b', 
        #                         np.where(p_arr_ori == 2, 'g',
        #                                 np.where(p_arr_ori == 3, 'darkorchid', 
        #                                         np.where(p_arr_ori == 4, 'saddlebrown', 
        #                                                 np.where(p_arr_ori == 5, 'goldenrod',
        #                                                             np.where(p_arr_ori == 6, 'black', 'tab:pink')))))))
        #     ori_list = x_ori[:max_links].reshape(n_ct, -1)
        #     overlap_colours_before = np.array([np.where(ori_list[:, i].sum() > 1, 'overlap', 'no') for i in range(MAX_ID_metabolites)])
        #     c_before[overlap_colours_before == 'overlap'] = 'tab:red'

        #     # c_before = np.where(i_before, c_before, 'tab:gray')
        #     c_after = np.where(p_arr_ov == 0, 'tab:gray', np.where(p_arr_ov == 1, 'b', 
        #                         np.where(p_arr_ov == 2, 'g',
        #                                 np.where(p_arr_ov == 3, 'darkorchid', 
        #                                         np.where(p_arr_ov == 4, 'saddlebrown', 
        #                                                 np.where(p_arr_ov == 5, 'goldenrod',
        #                                                             np.where(p_arr_ov == 6, 'black', 'tab:pink')))))))
        #     if k < 1:
        #         ov_list = x_ov[:max_links].reshape(n_ct, -1)
        #         overlap_colours_after = np.array([np.where(ov_list[:, i].sum() > 1, 'overlap', 'no') for i in range(MAX_ID_metabolites)])
        #         c_after[overlap_colours_after == 'overlap'] = 'k'

        #         ov_celltypes = np.where(ov_list[:, overlap_colours_after == 'overlap'])[0] + 1
        #         a_before = np.where(i_final_ori, 0.8, 0.1)
        #         a_after = np.where(i_final_ov,
        #                         np.where(np.isin(p_arr_ov, ov_celltypes.min()), 0.8, 0.2), 0.)
        #         a_after[overlap_colours_after == 'overlap'] = 0.8
        #                         # np.where(overlap_colours_after == 'overlap', 1, 0.2), 0.)

        #         s_after = np.where(overlap_colours_after == 'overlap', 80, 40)
        #         edgecolor_before = np.where(overlap_colours_before == 'overlap', 'k', c_before)
        #         edgecolor_after = np.where(overlap_colours_after == 'overlap', 'k', c_after)

        #     else:
        #         c_after = np.array(['tab:grey']*len(c_before))
        #         ov_list = x_ov[max_links:].reshape(n_ct, -1)
        #         overlap_colours_after = np.array([np.where(ov_list[:, i].sum() > 1, 'overlap', 'no') for i in range(MAX_ID_metabolites)])
        #         c_after[overlap_colours_after == 'overlap'] = 'tab:red'

        #         a_before = np.where(i_final_ori, 0.8, 0.1)
        #         a_after = np.where(i_final_ov,
        #                         np.where(overlap_colours_after == 'overlap', 1, 0.2), 0.)

        #         s_after = np.where(overlap_colours_after == 'overlap', 100, 40)
        #         edgecolor_before = np.where(overlap_colours_before == 'overlap', 'k', c_before)
        #         edgecolor_after = np.where(overlap_colours_after == 'overlap', 'k', c_after)

        #     fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3, 5))
        #     ax1.scatter(np.log10(met_pred_ori), np.log10(met_meas_ori), c=c_before, alpha=a_before, s=50, edgecolors=edgecolor_before)
        #     ax1.axline((-2, -2), (3, 3), c='k', transform=ax1.transAxes)
        #     ax1.text(1.02, 0.5, 'Original', size=SMALL_SIZE, rotation='vertical', va='center', transform=ax1.transAxes)
        #     # ax2.text(0.05, 0.8, r'$\chi_{excess}=$'+f'{init_residual[i_best_net][0].round(decimals=3):.2f}', transform=ax2.transAxes)
        #     # ax2.set_xlabel(r'$log_{10}\ Predicted\ metabolome$')
        #     ax1.set_ylabel(r'$log_{10}\ Data$', size=SMALL_SIZE)
        #     ax1.text(0.05, 0.9, f'RMSE = {mean_error_ori.round(decimals=3):.2f}', transform=ax1.transAxes)

        #     ax2.scatter(np.log10(met_pred_ov), np.log10(met_meas_ov), c=c_after, alpha=a_after, s=s_after, edgecolors=edgecolor_after)
        #     ax2.axline((-2, -2), (3, 3), c='k', transform=ax2.transAxes)
        #     ax2.text(1.02, 0.5, f'{ov_links_pooled[k]}', size=SMALL_SIZE, rotation='vertical', va='center', transform=ax2.transAxes)
        #     ax2.text(0.05, 0.9, f'RMSE = {mean_error_ov.round(decimals=3):.2f}', transform=ax2.transAxes)
        #     # ax2.text(0.05, 0.8, r'$\chi_{excess}=$'+f'{final_residual[i_best_net][0].round(decimals=3):.2f}', transform=ax3.transAxes)
        #     ax2.set_xlabel(r'$log_{10}\ Prediction$')
        #     ax2.set_ylabel(r'$log_{10}\ Data$', size=SMALL_SIZE)

        #     fig.suptitle(f'{linktype_pooled[k]}'+' overlap-'+f'{best_worst_pooled[k]}')
        #     fig.tight_layout(pad=0.6)
        #     if figsave_flag:
        #         fig.savefig(fig_path + '/'+f'{best_worst_pooled[k]}'+'-'+f'{linktype_pooled[k]}'+'-metabolome-prediction-comparison.png', dpi=300)
        #         plt.close(fig)
        #     else:
        #         fig.show()

# %%
"""Network optimisation simulations"""
############ Run network optimisation 'n_rep' times for a given cell line, each time starting with a new randomised network

for n_ct in range(5, 6):
    home_dir = os.getcwd() #+ '/codes'
    celltype_ID, celltypefreq, ec_metabolome_ID, ec_metabolome, met_baseline, core_mean = pd.read_pickle(home_dir + '/' + str(n_ct) + '-cells-data.pickle')
    cell_line_names = ec_metabolome.columns.to_numpy()

    slope_df, sublinear_cell_lines = pd.read_pickle(home_dir + '/cancer_power_law_stats.pickle')

    ### Filtering those networks for cell lines with power law slopes
    i_sublinear = np.where(np.isin(ec_metabolome.columns.values, sublinear_cell_lines))[0]
    ec_metabolome = ec_metabolome.iloc[:, i_sublinear]

    ######## Diet as the average of all the Baseline values
    diet = met_baseline.mean(axis=1)

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

    cellnum_init_all = cellnum_init_all[i_sublinear]
    cellnum_final_all = cellnum_final_all[i_sublinear]

    cell_line_names = ec_metabolome.columns.to_numpy()
    for i_cell_line in range(1):#range(len(sublinear_cell_lines[0])):
        cl = cell_line_names[i_cell_line]
        f = 0.5
        ec_real = ec_metabolome.loc[:, cl].values

        pickle_path = '../raw-output/'+str(n_ct)+'-celltypes/no-learn-balanced-net/'+cl
        [balance_arr_1ct, balance_arr_nct, balance_flag_list, balanced_networks_list,
                            rmse_arr_1ct, rmse_arr_nct,
                            ec_pred_arr_1ct, ec_pred_arr_nct,
                            index_arr_1ct, index_arr_nct,
                            prod_rates_1ct, prod_rates_nct,
                            production_ct_arr_1ct, production_ct_arr_nct] = pd.read_pickle(pickle_path + '/balanced-networks.pickle')
        all_networks = balanced_networks_list.copy()

        i_nonzero_celltypes = all_networks[0]['celltypes'].unique()
        i_nonzero_celltypes = np.sort(i_nonzero_celltypes)
        i_nonzero_celltypes = celltype_ID.values.copy()
        i_nonzero_metabolites = all_networks[0]['metabolites'].unique()

        MAX_ID_celltypes = len(i_nonzero_celltypes)  # MAX_ID_celltypes is the maximum of ID labels for celltypes.
        MAX_ID_metabolites = len(i_nonzero_metabolites)  # MAX_ID_metabolites is the maximum of ID labels for metabolites.

        cellnum_init = cellnum_init_all[i_cell_line]
        cellnum_final = cellnum_final_all[i_cell_line]
        bias = np.log10(ec_metabolome.iloc[:, i_cell_line].values + 1e-6) - np.log10(diet.values + 1e-6)

        reward_arr = np.array([0.2])
        penalty_arr = np.array([0.2])

        in_degree_flag = False

        for k in range(len(reward_arr)):
            x_optim_list = [[]]
            x_ori_list = [[]]
            x_all_list = [[]]
            x_ori_list_null = [[]]
            x_optim_list_null = [[]]
            error_list_all_reps = [[]]
            log_bias_list = [[]]
            log_bias_combined_list = [[]]
            log_bias_list_null = [[]]
            metabolome_pred_before_list = [[]]
            metabolome_meas_before_list = [[]]
            metabolome_pred_after_list = [[]]
            metabolome_meas_after_list = [[]]
            valid_index_before_list = [[]]
            valid_index_after_list = [[]]
            n_pred_list = [[]]
            residual_list = [[]]
            prod_overlap_list, con_overlap_list = [[]], [[]]

            i_balance = np.where(balance_flag_list * np.where(rmse_arr_nct <= 0.9, True, False))[0] # np.where(balance_flag_list)[0]
            n_reps = len(i_balance)
            for i in i_balance:
                all_params = np.array([f, cl,
                        cellnum_init, cellnum_final,
                        all_networks[i], diet, in_degree_flag,
                        MAX_ID_metabolites, MAX_ID_celltypes,
                        0.003, penalty_arr[k], reward_arr[k],
                        prod_rates_nct[i]], dtype=object)

                kT, penalty, reward, f, x_ori, x, x_list, elist, n_pred, residual, prod_overlap, con_overlap, met_pred_list, met_measured_list, i_list, bias, bias_combined, ewindow = run_network_optimisation(all_params)
    
                x_ori_list.append(x_ori)
                x_optim_list.append(x)
                x_all_list.append(x_list)
                error_list_all_reps.append(elist)
                log_bias_list.append(bias)
                log_bias_combined_list.append(bias_combined)
                n_pred_list.append(n_pred)
                residual_list.append(residual)
                prod_overlap_list.append(prod_overlap)
                con_overlap_list.append(con_overlap)
                metabolome_pred_before_list.append(met_pred_list[0])
                metabolome_pred_after_list.append(met_pred_list[-1])
                metabolome_meas_before_list.append(met_measured_list[0])
                metabolome_meas_after_list.append(met_measured_list[-1])
                valid_index_before_list.append(i_list[0])
                valid_index_after_list.append(i_list[-1])
                
                ### Null model for the same network as above
                Twindow = [len(ewindow)]
                print('Null sim length: '+str(Twindow))
                all_null_params = np.concatenate([all_params, Twindow])
                n_replicates = 1
                bias_null = []
                for i in range(n_replicates):
                    x_list, elist_null, n_pred_null, bias = run_null_optimisation(all_null_params)
                    bias_null.append(bias[-1])
                log_bias_list_null.append(bias_null)
                x_ori_list_null.append(x_list[0])
                x_optim_list_null.append(x_list[-1])



            x_ori_list = np.array(x_ori_list[1:])
            x_optim_list = np.array(x_optim_list[1:])
            x_all_list = np.array(x_all_list[1:], dtype=object)
            x_ori_list_null = np.array(x_ori_list_null[1:], dtype=object)
            x_optim_list_null = np.array(x_optim_list_null[1:], dtype=object)
            error_plot_list = np.array(error_list_all_reps[1:], dtype=object)
            log_bias_list = np.array(log_bias_list[1:], dtype=object)
            log_bias_combined_list = np.array(log_bias_combined_list[1:], dtype=object)
            log_bias_list_null = np.array(log_bias_list_null[1:], dtype=object)
            n_pred_list = np.array(n_pred_list[1:], dtype=object)
            residual_list = np.array(residual_list[1:], dtype=object)
            prod_overlap_list, con_overlap_list = np.array(prod_overlap_list[1:], dtype=object), np.array(con_overlap_list[1:], dtype=object)
            metabolome_pred_before_list = np.array(metabolome_pred_before_list[1:], dtype=object)
            metabolome_pred_after_list = np.array(metabolome_pred_after_list[1:], dtype=object)
            metabolome_meas_before_list = np.array(metabolome_meas_before_list[1:], dtype=object)
            metabolome_meas_after_list = np.array(metabolome_meas_after_list[1:], dtype=object)
            valid_index_before_list = np.array(valid_index_before_list[1:], dtype=object)
            valid_index_after_list = np.array(valid_index_after_list[1:], dtype=object)

            net_state = 'optim-net/'
            pickle_path = '../raw-output/'+str(n_ct)+'-celltypes/'+net_state+cl
            try:
                os.makedirs(pickle_path)
            except:
                pass

            pickle_out = open(pickle_path + "/reward-"+str(reward_arr[k])+"-penalty-"+str(penalty_arr[k])+"-optimised_network_output.pickle", "wb")
            #pickle.dump([net, i_selfish, i_intake, names], pickle_out)
            pickle.dump([x_all_list, x_ori_list, x_optim_list, error_plot_list, log_bias_list, log_bias_combined_list, log_bias_list_null, x_ori_list_null, x_optim_list_null,
                         n_pred_list, residual_list, balance_flag_list, prod_overlap_list, con_overlap_list,
                        metabolome_pred_before_list, metabolome_meas_before_list,
                        metabolome_pred_after_list, metabolome_meas_after_list,
                        valid_index_before_list, valid_index_after_list], pickle_out, protocol=2)
            pickle_out.close()


# %%
