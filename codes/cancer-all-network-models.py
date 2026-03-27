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
from scipy.sparse import csr_matrix
import pickle
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
import networkx as nx
from multiprocessing import Pool

import os
import numpy.matlib
# import all_plots
from tqdm import tqdm

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
    
    i_nonzero = np.where((ec_pred * ec_real) > 0, True, False)
    i_filt = np.where((m2b.sum(0) > 0), True, False)
    i_final = i_nonzero * i_filt
    diff = np.log10(ec_pred[i_final]) - np.log10(ec_real[i_final]) # / np.log10(ec_real[i_nonzero])
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
        ct_final = cellnum_max*celltypefreq.values
    
    ######## Compute matrices and predicted metabolome from learnt celltype abundances
    m2b, b2m, metabolome_pred_unfilt = calculate_metabolome_from_net(f, ct_final, diet, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
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

def generate_random_network(rnet, ec_real):
    # rnet = net.copy()
    # # n_unused = (rnet.iloc[:, -1] == 0).sum() # Number of unused metabolites
    # # n_edges = rnet.shape[0]
    # i_consumption_edges = np.where(bias_metabolome > 0)[0]
    # i_production_edges = np.where(bias_metabolome < 0)[0]

    # con_edges = np.random.choice([0, 2], len(i_consumption_edges), replace=True)
    # pro_edges = np.random.choice([0, 3], len(i_production_edges), replace=True)
    # rnet.iloc[i_consumption_edges, -1] = con_edges.copy()
    # rnet.iloc[i_production_edges, -1] = pro_edges.copy()

    # if MAX_ID_celltypes > 1:    
    #     assigned_edges = rnet.iloc[:len(bias_metabolome), -1].values
    #     shuffled_edges = np.repeat(assigned_edges, MAX_ID_celltypes-1)
    #     shuffled_edges = np.random.permutation(shuffled_edges)
        
    #     rnet.iloc[len(bias_metabolome):, -1] = shuffled_edges.copy()
        ## Production edges with partitioning

    quantiles = np.quantile(ec_real, np.linspace(0, 1, MAX_ID_celltypes+1)[1:-1])
    prod_celltypes = np.zeros_like(ec_real)
    for i in range(MAX_ID_celltypes-1):
        prod_celltypes = np.where(ec_real <= quantiles[i], prod_celltypes, prod_celltypes+1)
    prod_celltypes = np.int64(prod_celltypes + 1)

    # i_production = np.where(bias_metabolome < 0)[0]
    for i in range(1, MAX_ID_celltypes+1):
        # i_production = np.where(bias_metabolome < 0, True, False)
        i_ct_prod = np.where(prod_celltypes == i, True, False)
        n_pro = np.random.randint(1, np.max([i_ct_prod.sum(), 2]))
        i_final = np.random.choice(np.where(i_ct_prod)[0], n_pro, replace=True)
        pro_edges = np.random.choice([0, 3], len(i_final), replace=True)

        # i_consumption_edges = np.where(bias_metabolome > 0)[0]
        # con_edges = np.random.choice([0, 2], len(i_consumption_edges), replace=True)
        
        ct_index = np.where(rnet.iloc[:, 0]==i)[0]
        rnet.iloc[ct_index[i_final], -1] = pro_edges.copy()
        # rnet.iloc[ct_index[i_consumption_edges], -1] = con_edges.copy()
    
    pro_mets_final = rnet.iloc[np.where(rnet.iloc[:, -1]==3)[0], 2].values
    for i in range(1, MAX_ID_celltypes+1):
        ct_index = np.where(rnet.iloc[:, 0]==i)[0]
        n_con = np.random.randint(1, np.max([len(pro_mets_final), 2]))
        con_mets = np.random.choice(pro_mets_final, n_con, replace=True)
        i_con_edges = np.where(np.isin(rnet.iloc[ct_index, 2].values, con_mets))[0]
        rnet.iloc[ct_index[i_con_edges], -1] = np.where(rnet.iloc[ct_index[i_con_edges], -1] == 3, 5, 2)
    
    # con_mets_final = rnet.iloc[np.where(rnet.iloc[:, -1]==2)[0], 2].values
    # prod_mets_final = np.where(rnet.iloc[:, -1]==3)[0]
    # for i in prod_mets_final:
    #     if np.isin(rnet.iloc[i, 2], con_mets_final):
    #         continue
    #     else:
    #         rnet.iloc[i, -1] = 0

    return rnet

def generate_balance_part_network(net, bias_metabolome, ec_real):
    rnet = net.copy()
    rnet.iloc[:, -1] = 0
    # n_unused = (rnet.iloc[:, -1] == 0).sum() # Number of unused metabolites
    # n_edges = rnet.shape[0]

    rnet = generate_random_network(rnet, ec_real)
    balance = np.zeros(MAX_ID_celltypes)
    for i in range(1, MAX_ID_celltypes+1):
        net_temp = rnet[rnet['celltypes_ID']==i]
        i_consumed = np.where(np.isin(net_temp.iloc[:MAX_ID_metabolites, -1], [2, 5]))[0]
        i_produced = np.where(np.isin(net_temp.iloc[:MAX_ID_metabolites, -1], [3, 5]))[0]
        net_consumption = diet.values[i_consumed].sum()
        net_production = ec_real[i_produced].sum()
        balance[i-1] = net_production/net_consumption

    ## Balance calculation
    # b_low = 0.1
    # b_high = 1
    count = 0
    while (count <= 100) * (balance != 1).any():#((balance < b_low) + (balance > b_high)).any():
        balance_flag = False
        rnet = generate_random_network(rnet, ec_real)
        for i in range(1, MAX_ID_celltypes+1):
            net_temp = rnet[rnet['celltypes_ID']==i]
            i_consumed = np.where(np.isin(net_temp.iloc[:MAX_ID_metabolites, -1], [2, 5]))[0]
            i_produced = np.where(np.isin(net_temp.iloc[:MAX_ID_metabolites, -1], [3, 5]))[0]
            net_consumption = diet.values[i_consumed].sum()
            net_production = ec_real[i_produced].sum()
            balance[i-1] = net_production/net_consumption
        count += 1

    if count < 100: #((balance >= b_low) * (balance <= b_high)).all():
        balance_flag = True

    return rnet, balance_flag

####### Error function for GutCP-based algorithm
def pred_error_addingLinks(n_pred_init, residual_init, x, net_ori, f, col_name, diet, in_degree_flag, cellnum_init, cellnum_max, pred_params):
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
    ec_corr, ct_full, mean_error, metabolome_pred, metabolome_measured, i_final, bias_metabolome, n_pred_new = run_network_model(f, diet, col_name, cellnum_init, cellnum_max, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

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
    m2b_combined, b2m_combined, metabolome_pred_combined = calculate_metabolome_from_net(f, ct_final, diet, net_combined, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
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
    
    # # ## Strict interconversion
    # # n_inter = np.isin(i_produced, i_consumed).sum()
    # # n_non_inter = (~np.isin(i_produced, i_consumed)).sum()
    # inter_diff = n_inter - n_inter_old
    # non_inter_diff = n_non_inter - n_non_inter_old
    residual_diff = residual_new - residual_init

    penalty_param = pred_params['penalty']
    reward_param = pred_params['reward']
    pred_errorTotal = mean_error - np.where(n_pred_new > n_pred_init, reward_param, -penalty_param) #+ np.where(residual_new != 0, penalty_param * np.log10(residual_new), 0) #np.where(residual_init > residual_new, penalty_param, -reward_param) #- (inter_diff*reward_param) + (non_inter_diff*penalty_param)
    # + (penalty_param * n_non_inter) - (reward_param * n_inter)
    
    return [pred_errorTotal, metabolome_pred, metabolome_measured, i_final, bias_metabolome, mean_error, mean_error_combined, n_pred_new, residual_new]

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
    # current_step_list = []
    # pos_x_list = []
    # metID_list = []
    # celltypeID_list = []
    prior_list = []
    log_bias_list = []
    log_bias_combined_list = []
    metabolome_pred_list = []
    metabolome_measured_list = []
    valid_index_list = []
    n_pred_list = []
    residual_list = []

    ct0 = cellnum_init*celltypefreq.to_numpy()

    # Save original network features and prediction errors
    # net_ori, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(net_raw)

    m2b, b2m, met_pred = calculate_metabolome_from_net(f, ct0, diet, net_ori, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
    ####### Keep a record of the original network
    m2b_ori = (m2b!=0).astype(int).copy()
    b2m_ori = (b2m!=0).astype(int).copy()
    x_ori = np.concatenate([m2b_ori.flatten(), b2m_ori.flatten()]) # This is the adjacency matrix
    # fun = lambda x: pred_error_addingLinks(x, net_ori, f, cl, diet, in_degree_flag, cellnum_init, cellnum_final)
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

        # ## Strict interconversion
        # n_tot = len(i_consumed) + len(i_produced)
        # n_inter += np.intersect1d(i_consumed, i_produced).shape[0]#np.isin(i_produced, i_consumed).sum()
        # n_non_inter += n_tot - n_inter #(~np.isin(i_produced, i_consumed)).sum()

    residual_init = np.abs(balance - 1).sum() # This residual term will be zero when the balance becomes 1 for all celltypes

    error_before, metabolome_pred, metabolome_measured, i_final, bias_metabolome, log_bias_init, log_bias_combined_init, n_pred, residual_init = pred_error_addingLinks(n_pred_init, residual_init, x_ori, net_ori, f, cl, diet, in_degree_flag, cellnum_init, cellnum_final, pred_params)
    metabolome_pred_list.append(metabolome_pred)
    metabolome_measured_list.append(metabolome_measured)
    valid_index_list.append(i_final)
    log_bias_list.append(log_bias_init)
    log_bias_combined_list.append(log_bias_combined_init)
    n_pred_list.append(n_pred)
    residual_list.append(residual_init)
    # n_pred_before = len(np.where(metabolome_pred_before > 0)[0])
    bias_metabolome_ori = bias_metabolome.copy()

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
    Twindow = 750
    # numStepsNotAdded = 0
    # numAdditions = 0
    # numDeletions = 0
    error_window = []
    for i in range(30000):

        error_window.append(error_before)
        x_previous = x.copy()
        if np.random.uniform(0,1,1)[0] <= 0.5: # Each step chooses randomly between adding a new link or removing an existing link
            i_x = np.random.choice(np.where(x==0)[0], 1, p=prior_prob[x==0]/prior_prob[x==0].sum())[0] # New link is chosen based on the prior probability-smaller bias in prediction leads to smaller prior prob
            x[i_x] = 1
        elif (n_pred_init >= 4) * ((x==1).sum() > 0): # Only propose deletion if at least three metabolites can be non-trivially predicted after deletion
            i_x = np.random.choice(np.where(x==1)[0], 1, p=prior_prob[x==1]/prior_prob[x==1].sum())[0] # New link is chosen based on the prior probability-smaller bias in prediction leads to smaller prior prob
            x[i_x] = 0
        else:
            continue

        error_after, metabolome_pred, metabolome_measured, i_final, bias_metabolome, log_bias, log_bias_combined, n_pred, residual_after = pred_error_addingLinks(n_pred_init, residual_init, x, net_ori, f, cl, diet, in_degree_flag, cellnum_init, cellnum_final, pred_params) # Calculate prediction error with the modified network
        prior_prob = calculate_priors(bias_metabolome, prod_celltypes)

        if (n_pred >= 3) * (np.random.uniform(0,1,1)[0] < np.exp((error_before-error_after)/kT)): # If at least three metabolites are non-trivially predicted and the reduction in error is large enough, the proposed link addition/removal is accepted
            error_before = error_after
            # if x[i_x] == 1:
            #     # print('Addition accepted, error is ', error_before)
            #     numAdditions += 1
            # else:
            #     # print('Deletion accepted, error is ', error_before)
            #     numDeletions += 1 
            error_list.append(error_before)
            # current_step_list.append(i)
            # pos_x_list.append(i_x) # Record every link whose status has changed in the adjacency matrix
            prior_list.append(prior_prob[i_x])
            log_bias_list.append(log_bias)
            log_bias_combined_list.append(log_bias_combined)
            metabolome_pred_list.append(metabolome_pred)
            metabolome_measured_list.append(metabolome_measured)
            valid_index_list.append(i_final)
            n_pred_list.append(n_pred)
            residual_list.append(residual_after)
            n_pred_init = n_pred
            residual_init = residual_after.copy()
            # x_previous = x.copy()

            # if i_x < max_links:
            #     row_num = i_x // m2b_ori.shape[1]
            #     col_num = i_x - row_num * m2b_ori.shape[1]
            # elif i_x >= max_links:
            #     i_x = i_x - m2b_ori.shape[0] * m2b_ori.shape[1]
            #     row_num = i_x // b2m_ori.shape[1]
            #     col_num = i_x - row_num * b2m_ori.shape[1]
            # metID_list.append(row_num)
            # celltypeID_list.append(col_num)
            # numStepsNotAdded = 0
        else:  ## not accepted   
            x[i_x] = x_previous[i_x] # Maintain the previous state
            # numStepsNotAdded += 1
            n_pred_init = n_pred
            residual_init = residual_after.copy()
            # x_previous = x.copy()

        if (i > Twindow) and ((error_window[-1] - error_window[-Twindow]) > -(np.sqrt(Twindow)*kT)):
            break
    # n_pred_after = len(np.where(metabolome_pred_list[-1] > 0)[0])
    
    return [kT, pred_params['penalty'], pred_params['reward'], f, x_ori, x, error_list, n_pred_list, residual_list, metabolome_pred_list, metabolome_measured_list, valid_index_list, log_bias_list, log_bias_combined_list]#, consumption_added, production_added, consumption_deleted, production_deleted]

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

# %%
###### Running sensitivity simulations for kT, penalty and reward for one cell line, for different numbers of celltypes
for n_ct in range(1, 5):
    home_dir = os.getcwd()
    all_networks, i_intake, names = pd.read_pickle(home_dir + '/' + str(n_ct) + '-cells-cancer_network.pickle')
    # i_selfish = 0

    # pickle_in = open("data.pickle","rb")
    celltype_ID, celltypefreq, ec_metabolome_ID, ec_metabolome, met_baseline, core_mean = pd.read_pickle(home_dir + '/' + str(n_ct) + '-cells-data.pickle')

    slope_df, sublinear_cell_lines = pd.read_pickle(home_dir + '/cancer_power_law_stats.pickle')
    nets_temp = all_networks.copy()

    ### Filtering those networks for cell lines with power law slopes
    i_sublinear = np.where(np.isin(ec_metabolome.columns.values, sublinear_cell_lines))[0]
    all_networks = []
    for i in i_sublinear:
        all_networks.append(nets_temp[i])
    ec_metabolome = ec_metabolome.iloc[:, i_sublinear]

    i_nonzero_celltypes = all_networks[0]['celltypes_ID'].unique()
    i_nonzero_celltypes = np.sort(i_nonzero_celltypes)
    i_nonzero_celltypes = celltype_ID.values.copy()
    i_nonzero_metabolites = all_networks[0]['metabolites_ID'].unique()
    # i_nonzero_metabolites = np.sort(i_nonzero_metabolites)

    MAX_ID_celltypes = len(i_nonzero_celltypes)  # MAX_ID_celltypes is the maximum of ID labels for celltypes.
    MAX_ID_metabolites = len(i_nonzero_metabolites)  # MAX_ID_metabolites is the maximum of ID labels for metabolites.

    #### Model initialisation
    f_arr = np.linspace(0.1, 1, 5)
    cell_line_names = ec_metabolome.columns.values
    k = len(celltype_ID)  # Number of cell types in the model

    ######## Diet as the average of all the Baseline values
    diet = met_baseline.mean(axis=1)

    # ec_corr = np.zeros((len(f_arr), len(cell_line_names)))
    # ct_full = np.zeros((len(f_arr), len(cell_line_names))) # For one cell type
    # mean_error = np.zeros((len(f_arr), len(cell_line_names)))
    # # ct_full = np.zeros((len(f_arr), len(cell_line_names), k)) # For more than one cell type
    # metabolome_pred = np.zeros((len(f_arr), len(cell_line_names), len(ec_metabolome)))
    # slopes = np.zeros_like(ec_corr)
    # intercepts = np.zeros_like(ec_corr)


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

    ####### Sensitivity plots
    n_reps = 50
    bias = np.log10(ec_metabolome.iloc[:, 0].values + 1e-6) - np.log10(diet.values + 1e-6)

    i_cell_line = np.where(cell_line_names == 'A549-ATCC')[0][0]
    net_raw = generate_random_network(all_random_networks[i_cell_line], bias)

    all_params = np.array([0.5, cell_line_names[i_cell_line],
                cellnum_init_all[i_cell_line], cellnum_final_all[i_cell_line],
                net_raw, diet, in_degree_flag,
                MAX_ID_metabolites, MAX_ID_celltypes,
                0.02, 0., 0.], dtype=object)

    #### Sensitivity to kT-no penalty or reward
    kT_arr = np.repeat(np.array([1e-03, 0.5*1e-2, 1e-2, 0.5*1e-1, 1e-1]), n_reps)
    iter_params_list_kt = np.repeat(all_params[np.newaxis, :], len(kT_arr), 0)
    iter_params_list_kt[:, -3] = kT_arr.copy()

    #### Sensitivity to penalty-no reward and kT=0.0003
    penalty_arr = np.repeat(np.array([1e-4, 0.5*1e-3, 1e-3, 0.5*1e-2, 1e-2]), n_reps)
    iter_params_list_pn = np.repeat(all_params[np.newaxis, :], len(penalty_arr), 0)
    iter_params_list_pn[:, -2] = penalty_arr.copy()

    #### Sensitivity to reward-no penalty and kT=0.0003
    reward_arr = np.repeat(np.array([1e-4, 0.5*1e-3, 1e-3, 0.5*1e-2, 1e-2]), n_reps)
    iter_params_list_rw = np.repeat(all_params[np.newaxis, :], len(reward_arr), 0)
    iter_params_list_rw[:, -1] = reward_arr.copy()

    kt_vals = []
    kt_errors = []
    for param in tqdm(iter_params_list_kt, desc='kT'):
        data = run_network_optimisation(param)
        kt_vals.append(data[0])
        kt_errors.append(data[-1][-1])

    penalty_vals = []
    penalty_errors = []
    for param in tqdm(iter_params_list_pn, desc='penalty'):
        data = run_network_optimisation(param)
        penalty_vals.append(data[1])
        penalty_errors.append(data[-1][-1])

    reward_vals = []
    reward_errors = []
    for param in tqdm(iter_params_list_rw, desc='reward'):
        data = run_network_optimisation(param)
        reward_vals.append(data[2])
        reward_errors.append(data[-1][-1])

    ##### Byproduct fraction vs celltype number
    f_arr = np.repeat(np.linspace(0.1, 1, 5), n_reps)
    all_params = np.array([0., cell_line_names[0],
                cellnum_init_all[0], cellnum_final_all[0],
                net_raw, diet, in_degree_flag,
                MAX_ID_metabolites, MAX_ID_celltypes,
                0.02, 0., 0.], dtype=object)
    iter_params_list = np.repeat(all_params[np.newaxis, :], len(f_arr), 0)
    iter_params_list[:, 0] = f_arr.copy()

    f_vals = []
    f_errors = []
    n_pred_before = []
    n_pred_after = []

    for params in tqdm(iter_params_list, desc='f'):
        f_data = run_network_optimisation(params)
        f_vals.append(f_data[3])
        f_errors.append(f_data[-1][-1])

    # if __name__ == '__main__':
    #     __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    #     pool = Pool(8)
    #     f_data = pool.map(run_network_optimisation, iter_params_list) #iterate over combinations
    #     pool.close()
    #     pool.join()

    # f_errors = np.array([arr[3] for arr in f_data])

    # if __name__ == '__main__':
    #     __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    #     pool = Pool(8)
    #     kt_data = pool.map(run_network_optimisation, iter_params_list_kt) #iterate over combinations
    #     pool.close()
    #     pool.join()

    #     __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    #     pool = Pool(8)
    #     penalty_data = pool.map(run_network_optimisation, iter_params_list_pn) #iterate over combinations
    #     pool.close()
    #     pool.join()
        
    #     __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    #     pool = Pool(8)
    #     reward_data = pool.map(run_network_optimisation, iter_params_list_rw) #iterate over combinations
    #     pool.close()
    #     pool.join()

    # kt_errors = np.array([arr[3] for arr in kt_errors])
    # penalty_errors = np.array([arr[3] for arr in penalty_errors])
    # reward_errors = np.array([arr[3] for arr in reward_errors])
    kt_df = pd.DataFrame({'kT': kt_vals, 'rmse': kt_errors})
    penalty_df = pd.DataFrame({'Penalty': penalty_vals, 'rmse': penalty_errors})
    reward_df = pd.DataFrame({'Reward': reward_vals, 'rmse': reward_errors})
    f_df = pd.DataFrame({'f': f_arr, 'rmse': f_errors}).groupby('f').agg(('mean', 'std'))

    net_state = 'optim-net/'
    pickle_path = '../raw-output/'+str(k)+'-celltypes/'+net_state+cell_line_names[i_cell_line]
    try:
        os.makedirs(pickle_path)
    except:
        pass

    pickle_out = open(pickle_path + "/sensitivity-kt-penalty-reward.pickle", "wb")
    pickle.dump([kt_df, penalty_df, reward_df], pickle_out, protocol=2)
    pickle_out.close()

    pickle_out = open(pickle_path + "/by-product-fraction.pickle", "wb")
    pickle.dump([f_df], pickle_out, protocol=2)
    pickle_out.close()

# %%
pickle_path = '../raw-output/'+str(1)+'-celltypes/'+net_state+cell_line_names[i_cell_line]
kt_df1, pn_df1, rw_df1 = pd.read_pickle(pickle_path + '/sensitivity-kt-penalty-reward.pickle')
f_df1 = pd.read_pickle(pickle_path + "/by-product-fraction.pickle")

pickle_path = '../raw-output/'+str(2)+'-celltypes/'+net_state+cell_line_names[i_cell_line]
kt_df2, pn_df2, rw_df2 = pd.read_pickle(pickle_path + '/sensitivity-kt-penalty-reward.pickle')
f_df2 = pd.read_pickle(pickle_path + "/by-product-fraction.pickle")

pickle_path = '../raw-output/'+str(3)+'-celltypes/'+net_state+cell_line_names[i_cell_line]
kt_df3, pn_df3, rw_df3 = pd.read_pickle(pickle_path + '/sensitivity-kt-penalty-reward.pickle')
f_df3 = pd.read_pickle(pickle_path + "/by-product-fraction.pickle")

pickle_path = '../raw-output/'+str(4)+'-celltypes/'+net_state+cell_line_names[i_cell_line]
kt_df4, pn_df4, rw_df4 = pd.read_pickle(pickle_path + '/sensitivity-kt-penalty-reward.pickle')
f_df4 = pd.read_pickle(pickle_path + "/by-product-fraction.pickle")

# n_reps = 50
# kT_arr = np.repeat(np.array([1e-5, 0.5*1e-4, 1e-4, 0.5*1e-3, 1e-3]), n_reps)
# penalty_arr = np.repeat(np.array([1e-5, 0.5*1e-4, 1e-4, 0.5*1e-3, 1e-3]), n_reps)
# reward_arr = np.repeat(np.array([1e-5, 0.5*1e-4, 1e-4, 0.5*1e-3, 1e-3]), n_reps)
figsave_flag = True
ct_num = np.array([np.repeat(1, len(kt_df1)), np.repeat(2, len(kt_df2)), np.repeat(3, len(kt_df3)), np.repeat(4, len(kt_df4))]).ravel()

kt_df_all = pd.DataFrame({'kT': np.concatenate([kT_arr, kT_arr, kT_arr, kT_arr]),
                'CTNum': ct_num,
                'MeanError': np.concatenate([kt_df1.loc[:, 'rmse'].values,
                                             kt_df2.loc[:, 'rmse'].values,
                                             kt_df3.loc[:, 'rmse'].values,
                                             kt_df4.loc[:, 'rmse'].values])})
sns.lineplot(data=kt_df_all, x='CTNum', hue='kT', y='MeanError', palette='crest')
if figsave_flag:
    plt.savefig('../figures/sensitivity-plot-rmse-vs-kt.png', dpi=300)
    plt.close()

pn_df_all = pd.DataFrame({'Penalty': np.concatenate([penalty_arr, penalty_arr, penalty_arr, penalty_arr]),
                'CTNum': ct_num,
                'MeanError': np.concatenate([pn_df1.loc[:, 'rmse'].values,
                                             pn_df2.loc[:, 'rmse'].values,
                                             pn_df3.loc[:, 'rmse'].values,
                                             pn_df4.loc[:, 'rmse'].values])})
sns.lineplot(data=pn_df_all, x='CTNum', hue='Penalty', y='MeanError', palette='crest')
if figsave_flag:
    plt.savefig('../figures/sensitivity-plot-rmse-vs-penalty.png', dpi=300)
    plt.close()

rw_df_all = pd.DataFrame({'Reward': np.concatenate([reward_arr, reward_arr, reward_arr, reward_arr]),
                'CTNum': ct_num,
                'MeanError': np.concatenate([rw_df1.loc[:, 'rmse'].values,
                                             rw_df2.loc[:, 'rmse'].values,
                                             rw_df3.loc[:, 'rmse'].values,
                                             rw_df4.loc[:, 'rmse'].values])})
sns.lineplot(data=rw_df_all, x='CTNum', hue='Reward', y='MeanError', palette='crest')
if figsave_flag:
    plt.savefig('../figures/sensitivity-plot-rmse-vs-reward.png', dpi=300)
    plt.close()

# %%
heatmap_df = pd.DataFrame(np.array([f_df4[0].iloc[:, 0],
                                    f_df3[0].iloc[:, 0],
                                    f_df2[0].iloc[:, 0],
                                    f_df1[0].iloc[:, 0]]),
                          columns=np.linspace(0.1, 1, 5), index=[4, 3, 2, 1])
plt.figure(figsize=(9, 6))
sns.heatmap(data=heatmap_df, cmap='crest',
            cbar_kws={'label': 'RMSE'})
plt.xlabel(r'Byproduct fraction, $f$')
plt.ylabel(r'Celltype number')
if figsave_flag:
    plt.savefig('../figures/heatmap-byproduct-fraction-celltype-number-vs-rmse.png', dpi=300)
    plt.close()

rmse_df= pd.DataFrame(np.array([f_df1[0].iloc[:, 0],
                                f_df2[0].iloc[:, 0],
                                f_df3[0].iloc[:, 0],
                                f_df4[0].iloc[:, 0]]),
                          columns=np.linspace(0.1, 1, 5), index=[1, 2, 3, 4]).melt(var_name='f', value_name='mean', ignore_index=False).reset_index(names='Celltype number')
rmse_df.loc[:, 'sd'] = np.array([f_df1[0].iloc[:, 1],
                                    f_df2[0].iloc[:, 1],
                                    f_df3[0].iloc[:, 1],
                                    f_df4[0].iloc[:, 1]]).T.ravel()

rmse_df.groupby(['f']).plot(x='Celltype number', y='mean', yerr='sd',
                            kind='bar', capsize=3, figsize=(5, 4), legend=False,
                            ylabel=r'RMSE', xlabel=r'Celltype number')
# if figsave_flag:
#     plt.savefig('../figures/fraction-celltype-number-vs-rmse.png', dpi=300)
#     plt.close()


# %%
########### Sensitivity for reward-penalty combinations
home_dir = os.getcwd()
n_ct = 2
all_networks, i_intake, names = pd.read_pickle(home_dir + '/' + str(n_ct) + '-cells-cancer_network.pickle')
# i_selfish = 0

# pickle_in = open("data.pickle","rb")
celltype_ID, celltypefreq, ec_metabolome_ID, ec_metabolome, met_baseline, core_mean = pd.read_pickle(home_dir + '/' + str(n_ct) + '-cells-data.pickle')

slope_df, sublinear_cell_lines = pd.read_pickle(home_dir + '/cancer_power_law_stats.pickle')
nets_temp = all_networks.copy()

### Filtering those networks for cell lines with power law slopes
i_sublinear = np.where(np.isin(ec_metabolome.columns.values, sublinear_cell_lines))[0]
all_networks = []
for i in i_sublinear:
    all_networks.append(nets_temp[i])
ec_metabolome = ec_metabolome.iloc[:, i_sublinear]

i_nonzero_celltypes = all_networks[0]['celltypes_ID'].unique()
i_nonzero_celltypes = np.sort(i_nonzero_celltypes)
i_nonzero_celltypes = celltype_ID.values.copy()
i_nonzero_metabolites = all_networks[0]['metabolites_ID'].unique()

MAX_ID_celltypes = len(i_nonzero_celltypes)  # MAX_ID_celltypes is the maximum of ID labels for celltypes.
MAX_ID_metabolites = len(i_nonzero_metabolites)  # MAX_ID_metabolites is the maximum of ID labels for metabolites.

n_reps = 50
bias = np.log10(ec_metabolome.iloc[:, 0].values + 1e-6) - np.log10(diet.values + 1e-6)

reward_arr = np.repeat(np.array([0, 1e-3, 1e-2]), n_reps)
reward_arr = np.repeat(reward_arr, 3)
penalty_arr = np.repeat(np.array([0, 1e-3, 1e-2]), n_reps)
penalty_arr = np.repeat(penalty_arr[np.newaxis, :], 3, axis=0).ravel()

reward_vals = []
penalty_vals = []
errors = []
n_pred_before = []
n_pred_after = []
for i in tqdm(range(len(penalty_arr))):
    net_raw = generate_random_network(all_networks[i_cell_line], bias)
    all_params = np.array([0.3, cell_line_names[i_cell_line],
              cellnum_init_all[i_cell_line], cellnum_final_all[i_cell_line],
              net_raw, diet, in_degree_flag,
              MAX_ID_metabolites, MAX_ID_celltypes,
              0.003, penalty_arr[i], reward_arr[i]], dtype=object)
    
    all_data = run_network_optimisation(all_params)
    reward_vals.append(all_data[2])
    penalty_vals.append(all_data[1])
    errors.append(all_data[-1][-1])
    n_pred_before.append(all_data[7][0])
    n_pred_after.append(all_data[7][-1])

# reward_test_df = pd.DataFrame({'reward': reward_vals,
#                                'rmse': reward_errors,
#                                'Original': n_pred_before,
#                                'Optimised': n_pred_after})

# penalty_test_df = pd.DataFrame({'penalty': penalty_vals,
#                                'rmse': reward_errors,
#                                'Original': n_pred_before,
#                                'Optimised': n_pred_after})

test_df = pd.DataFrame({'penalty': penalty_vals,
                        'reward': reward_vals,
                        'rmse': errors,
                        'Original': n_pred_before,
                        'Optimised': n_pred_after})
test_df.loc[:, 'PredChange'] = test_df.loc[:, 'Optimised'] - test_df.loc[:, 'Original']


fig_path = '../figures/'+str(n_ct)+'-celltypes/'+net_state+cell_line_names[i_cell_line]
try:
    os.makedirs(fig_path)
except:
    pass

sns.set_style('darkgrid')
fig = sns.catplot(data=test_df, x='penalty', y='PredChange',
            kind='box', col='reward', height=5, aspect=0.8,
            whis=(5, 95), fliersize=0.)
fig.refline(y=0, color='tab:red', linestyle='dashed', linewidth=3)
fig.set_axis_labels('Penalty', 'Change in # metabolites predicted', fontsize=17)
plt.savefig(fig_path + '/sensitivity-penalty-cross-reward-npred.png', dpi=300)

# reward_test_df = reward_test_df.melt(id_vars=['reward', 'rmse'], value_vars=['Original', 'Optimised'], value_name='NumPred', var_name='NetType')

# penalty_test_df = penalty_test_df.melt(id_vars=['penalty', 'rmse'], value_vars=['Original', 'Optimised'], value_name='NumPred', var_name='NetType')

# sns.boxplot(data=penalty_test_df, x='penalty', y='NumPred',
#             hue='NetType', palette='crest',
#             whis=(5, 95), fliersize=0.)
# plt.xlabel('Penalty')
# plt.ylabel('# metabolites predicted')
# plt.legend(title='Network type', ncols=3)
# plt.savefig('../figures/sensitivity-penalty-vs-npred.png', dpi=300)

# sns.boxplot(data=reward_test_df, x='reward', y='NumPred',
#             hue='NetType', palette='crest',
#             whis=(5, 95), fliersize=0.)
# plt.xlabel('Reward')
# plt.ylabel('# metabolites predicted')
# plt.legend(title='Network type', ncols=3)
# plt.savefig('../figures/sensitivity-reward-vs-npred.png', dpi=300)



# %%
"""Network optimisation simulations"""
############ Run network optimisation 'n_rep' times for a given cell line, each time starting with a new randomised network
for n_ct in tqdm(range(2, 3), desc='n_ct'):
    home_dir = os.getcwd() #+ '/codes'
    # n_ct = 1
    # all_networks, i_intake, names = pd.read_pickle(home_dir + '/' + str(n_ct) + '-cells-cancer_network.pickle')
    # i_selfish = 0

    # pickle_in = open("data.pickle","rb")
    celltype_ID, celltypefreq, ec_metabolome_ID, ec_metabolome, met_baseline, core_mean = pd.read_pickle(home_dir + '/' + str(n_ct) + '-cells-data.pickle')
    cell_line_names = ec_metabolome.columns.to_numpy()

    slope_df, sublinear_cell_lines = pd.read_pickle(home_dir + '/cancer_power_law_stats.pickle')
    # nets_temp = all_networks.copy()

    ### Filtering those networks for cell lines with power law slopes
    i_sublinear = np.where(np.isin(ec_metabolome.columns.values, sublinear_cell_lines))[0]
    # all_networks = []
    # for i in i_sublinear:
    #     all_networks.append(nets_temp[i])
    ec_metabolome = ec_metabolome.iloc[:, i_sublinear]

    cl = sublinear_cell_lines[0]
    pickle_path = '../raw-output/'+str(n_ct)+'-celltypes/no-learn-balanced-net/'+cl
    [balance_arr_1ct, balance_arr_nct, balance_flag_list, balanced_networks_list,
                            rmse_arr_1ct, rmse_arr_nct,
                            ec_pred_arr_1ct, ec_pred_arr_nct,
                            index_arr_1ct, index_arr_nct,
                            production_ct_arr_1ct, production_ct_arr_nct] = pd.read_pickle(pickle_path + '/balanced-networks.pickle')
    all_networks = balanced_networks_list.copy()

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

    i_nonzero_celltypes = all_networks[0]['celltypes'].unique()
    i_nonzero_celltypes = np.sort(i_nonzero_celltypes)
    i_nonzero_celltypes = celltype_ID.values.copy()
    i_nonzero_metabolites = all_networks[0]['metabolites'].unique()

    MAX_ID_celltypes = len(i_nonzero_celltypes)  # MAX_ID_celltypes is the maximum of ID labels for celltypes.
    MAX_ID_metabolites = len(i_nonzero_metabolites)  # MAX_ID_metabolites is the maximum of ID labels for metabolites.

    cell_line_names = ec_metabolome.columns.to_numpy()
    for i_cell_line in tqdm(range(len(cell_line_names[:1])), desc='Cell lines'):
        # i_cell_line = np.where(cell_line_names == 'A549-ATCC')[0][0]
        cl = cell_line_names[i_cell_line]
        f = 0.5
        ec_real = ec_metabolome.loc[:, cl].values

        cellnum_init = cellnum_init_all[i_cell_line]
        cellnum_final = cellnum_final_all[i_cell_line]
        bias = np.log10(ec_metabolome.iloc[:, i_cell_line].values + 1e-6) - np.log10(diet.values + 1e-6)

        reward_arr = np.array([0.1])
        penalty_arr = np.array([0.1])

        in_degree_flag = False

        for k in range(len(reward_arr)):
            x_optim_list = [[]]
            x_ori_list = [[]]
            error_list_all_reps = [[]]
            log_bias_list = [[]]
            log_bias_combined_list = [[]]
            metabolome_pred_before_list = [[]]
            metabolome_meas_before_list = [[]]
            metabolome_pred_after_list = [[]]
            metabolome_meas_after_list = [[]]
            valid_index_before_list = [[]]
            valid_index_after_list = [[]]
            n_pred_list = [[]]
            residual_list = [[]]
            # balance_flag_list = []

            n_reps = len(all_networks)
            for i in np.arange(n_reps):
                # net_raw = generate_random_network(all_networks[i_cell_line], bias)
                # net_raw_balanced, balance_flag = generate_balance_part_network(all_networks[i_cell_line], bias, ec_real)
                # balance_flag_list.append(balance_flag)

                # net_ori, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(all_networks[i])
                all_params = np.array([f, cl,
                        cellnum_init, cellnum_final,
                        all_networks[i], diet, in_degree_flag,
                        MAX_ID_metabolites, MAX_ID_celltypes,
                        0.003, penalty_arr[k], reward_arr[k]], dtype=object)

                kT, penalty, reward, f, x_ori, x, elist, n_pred, residual, met_pred_list, met_measured_list, i_list, bias, bias_combined = run_network_optimisation(all_params)

                x_ori_list.append(x_ori)
                x_optim_list.append(x)
                error_list_all_reps.append(elist)
                log_bias_list.append(bias)
                log_bias_combined_list.append(bias_combined)
                n_pred_list.append(n_pred)
                residual_list.append(residual)
                metabolome_pred_before_list.append(met_pred_list[0])
                metabolome_pred_after_list.append(met_pred_list[-1])
                metabolome_meas_before_list.append(met_measured_list[0])
                metabolome_meas_after_list.append(met_measured_list[-1])
                valid_index_before_list.append(i_list[0])
                valid_index_after_list.append(i_list[-1])

                # print('Round', i+1, ', initial rmse is', log_bias[0])
                # print('Network optmisation ended with final rmse', log_bias[-1])
                # print(n_pred[0], 'metabolites predicted initially and', n_pred[-1], 'after optimisation')
                # print('------------------')

            x_ori_list = np.array(x_ori_list[1:])
            x_optim_list = np.array(x_optim_list[1:])
            error_plot_list = np.array(error_list_all_reps[1:], dtype=object)
            log_bias_list = np.array(log_bias_list[1:], dtype=object)
            log_bias_combined_list = np.array(log_bias_combined_list[1:], dtype=object)
            n_pred_list = np.array(n_pred_list[1:], dtype=object)
            residual_list = np.array(residual_list[1:], dtype=object)
            metabolome_pred_before_list = np.array(metabolome_pred_before_list[1:], dtype=object)
            metabolome_pred_after_list = np.array(metabolome_pred_after_list[1:], dtype=object)
            metabolome_meas_before_list = np.array(metabolome_meas_before_list[1:], dtype=object)
            metabolome_meas_after_list = np.array(metabolome_meas_after_list[1:], dtype=object)
            valid_index_before_list = np.array(valid_index_before_list[1:], dtype=object)
            valid_index_after_list = np.array(valid_index_after_list[1:], dtype=object)

            net_state = 'optim-net/balance-partition/'
            pickle_path = '../raw-output/'+str(n_ct)+'-celltypes/'+net_state+cl
            try:
                os.makedirs(pickle_path)
            except:
                pass

            pickle_out = open(pickle_path + "/reward-"+str(all_params[-1])+"-penalty-"+str(all_params[-2])+"-optimised_network_output.pickle", "wb")
            #pickle.dump([net, i_selfish, i_intake, names], pickle_out)
            pickle.dump([x_ori_list, x_optim_list, error_plot_list, log_bias_list, log_bias_combined_list, n_pred_list, residual_list, balance_flag_list,
                        metabolome_pred_before_list, metabolome_meas_before_list,
                        metabolome_pred_after_list, metabolome_meas_after_list,
                        valid_index_before_list, valid_index_after_list], pickle_out, protocol=2)
            pickle_out.close()

# %%
##### Visualising output
net_state = 'optim-net/'
figsave_flag = 1
fig_path = '../figures/2-celltypes/'+net_state+cl
try:
    os.makedirs(fig_path)
except:
    pass

##### I'm calling this the general summary, whatever that means
# f, ax = plt.subplots(2, 2, figsize=(8, 5))
fig = plt.figure(figsize=(9.5, 6))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])

#### A quick glance of where sims have begun and ended
colors = np.where(balance_flag_list, 'b', 'tab:red')
for i in range(n_reps):
    sns.lineplot(log_bias_list[i], ax=ax1, color=colors[i])
ax1.set_xlabel("Add/remove steps")
ax1.set_ylabel("RMSE")
ax1.set_title("%d replicate runs" % n_reps)


#### Pairwise comparison of combined vs uncombined network predictions
final_error = np.array([arr[-1] for arr in log_bias_list])
final_error_combined = np.array([arr[-1] for arr in log_bias_combined_list])
df = pd.DataFrame({'N_CT': np.concatenate([np.ones_like(final_error), np.ones_like(final_error)+1]).astype(int),
                    'Replicate': np.concatenate([np.arange(1, 1+n_reps), np.arange(1, 1+n_reps)]),
                    'Balance': np.concatenate([balance_flag_list, balance_flag_list]),
                    'RMSE': np.concatenate([final_error_combined, final_error])})
sns.lineplot(data=df, x='N_CT', y='RMSE',
            units='Replicate',
            hue='Balance', palette='coolwarm_r', hue_norm=(0, 1),
            dashes=False, estimator=None, ax=ax2, legend=True)
sns.scatterplot(data=df, x='N_CT', y='RMSE',
                hue='Balance', palette='coolwarm_r', hue_norm=(0, 1),
                edgecolor='face', alpha=0.9, ax=ax2, legend=False)
ax2.set_xlabel(r'$N_{CT}$')
ax2.set(xlim=(0.5, 2.5), xticks=[1, 2])
# ax2.get_legend().remove()

#### More non-trivially predicted metabolites means more error?
num_pred = np.array([arr[-1] for arr in n_pred_list])
final_error = np.array([arr[-1] for arr in log_bias_list])
sns.regplot(x=num_pred, y=final_error, color='k', ax = ax3,
            line_kws={'color': 'r', 'linewidth': 2}, scatter_kws={'s': 10})
rsq, pval = spearmanr(num_pred, final_error)
ax3.text(0.05, 0.9, f'rho = {rsq**2:.2f}, $p$ = {pval:.2f}', transform=ax3.transAxes)
ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax[1, 0].scatter(initial_error, final_error, c='k', s=5)
ax3.set_xlabel('# metabolites predicted')
ax3.set_ylabel('Final RMSE')

# for i in range(n_reps):
#     sns.lineplot(log_bias_list[i], ax=ax[1, 1])
# ax[1, 1].set_xlabel("Add/remove steps")
# ax[1, 1].set_ylabel("Met_RMSE")
# ax[1, 1].set_title("%d replicate runs" % n_reps)
# sns.regplot(x=initial_error, y=sim_length, color='k', ax = ax[1, 1],
#             line_kws={'color': 'r', 'linewidth': 2}, scatter_kws={'s': 10})
# rsq, pval = pearsonr(initial_error, sim_length)
# ax[1, 1].text(0.05, 0.9, f'$r^2$ = {rsq**2:.2f}, $p$ = {pval:.2f}', transform=ax[1, 1].transAxes)
# # ax[1, 0].scatter(initial_error, final_error, c='k', s=5)
# ax[1, 1].set_xlabel('Initial prediction error')
# ax[1, 1].set_ylabel('Step number')

fig.suptitle(f'Network optimisation for {cl} with reward {reward_arr[0]}')
fig.tight_layout()
if figsave_flag:
    fig.savefig(fig_path+'/with-balance-all-replicates-summary.png', dpi=300)
    print("Figure saved at "+fig_path)
    plt.close(fig)


# %%
##### Check out the best performing network
pred_error_change = np.array([list[0]-list[-1] for list in log_bias_list])
init_pred_error = np.array([arr[0] for arr in log_bias_list])
final_pred_error = np.array([arr[-1] for arr in log_bias_list])

init_residual = np.array([arr[0] for arr in residual_list])
final_residual = np.array([arr[-1] for arr in residual_list])

i_best_net = np.where(final_pred_error == final_pred_error.min())[0]#np.where(pred_error_change == pred_error_change.max())[0]

met_pred_before = metabolome_pred_before_list[i_best_net][0].astype(np.float64)
met_pred_after = metabolome_pred_after_list[i_best_net][0].astype(np.float64)

met_meas_before = metabolome_meas_before_list[i_best_net][0].astype(np.float64)
met_meas_after = metabolome_meas_after_list[i_best_net][0].astype(np.float64)

i_before = valid_index_before_list[i_best_net][0].astype(bool)
i_after = valid_index_after_list[i_best_net][0].astype(bool)

######## Convert x to net structure (convert the adjacency matrix into the edge list)

x_ori_best = x_ori_list[i_best_net].flatten()
x_optim_best = x_optim_list[i_best_net].flatten()

net_ori = net_from_x(x_ori_best, MAX_ID_metabolites, MAX_ID_celltypes)
net_optim = net_from_x(x_optim_best, MAX_ID_metabolites, MAX_ID_celltypes)

# %%
######### Change in error with additions and deletions
f = 0.5#f_arr[0]
ec_corr_old, ct_full_old, mean_error_old, metabolome_pred_old, metabolome_measured_old, i_final_old, bias_metabolome_old, n_predicted_old = run_network_model(f, diet, cl, cellnum_init_all[0], cellnum_final_all[0], net_ori, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

ec_corr, ct_full, mean_error, metabolome_pred, metabolome_measured, i_final, bias_metabolome, n_predicted = run_network_model(f, diet, cl, cellnum_init_all[0], cellnum_final_all[0], net_optim, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

print('Metabolome deviation with old network is: ', mean_error_old)
print('Metabolome deviation with improved network is: ', mean_error)
print('------------------------------------------------------------------------')

max_links = MAX_ID_metabolites * MAX_ID_celltypes
p_arr_ori = np.array([i*j for i, j in zip(x_ori_best[max_links:].reshape(2, -1), [1, 2])]).sum(0)
p_arr_optim = np.array([i*j for i, j in zip(x_optim_best[max_links:].reshape(2, -1), [1, 2])]).sum(0)

c_before = np.where(p_arr_ori == 0, 'tab:gray', 
                    np.where(p_arr_ori == 1, 'tab:blue', 
                             np.where(p_arr_ori == 2, 'tab:green', 'tab:red')))
c_after = np.where(p_arr_optim == 0, 'tab:gray', 
                   np.where(p_arr_optim == 1, 'tab:blue', 
                            np.where(p_arr_optim == 2, 'tab:green', 'tab:red')))

a_before = np.where(i_before, 0.8, 0.25)
a_after = np.where(i_after, 0.8, 0.25)

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(7, 4))
ax[0].scatter(np.log10(met_pred_before), np.log10(met_meas_before), c=c_before, alpha=a_before, s=30)
# ax[0].scatter(np.log10(metabolome_pred_old+1e-7), np.log10(metabolome_measured_old+1e-7), c='k', s=9)
ax[0].axline((-2, -2), (3, 3), c='k')
ax[0].set_title('Old network')
ax[0].text(0.05, 0.9, f'RMSE = {init_pred_error[i_best_net][0].round(decimals=3):.2f}', transform=ax[0].transAxes)
ax[0].text(0.05, 0.8, r'$\chi_{excess}=$'+f'{init_residual[i_best_net][0].round(decimals=3):.2f}', transform=ax[0].transAxes)
# ax[0].set_xlabel(r'$log_{10}\ Predicted\ metabolome$')
ax[0].set_ylabel(r'$log_{10}\ Empirical\ data$')

ax[1].scatter(np.log10(met_pred_after), np.log10(met_meas_after), c=c_after, alpha=a_after, s=30)
# ax[1].scatter(np.log10(metabolome_pred+1e-7), np.log10(metabolome_measured+1e-7), c='k', s=9)
ax[1].axline((-2, -2), (2, 2), c='k')
ax[1].set_title('New network')
ax[1].text(0.05, 0.9, f'RMSE = {final_pred_error[i_best_net][0].round(decimals=3):.2f}', transform=ax[1].transAxes)
ax[1].text(0.05, 0.8, r'$\chi_{excess}=$'+f'{final_residual[i_best_net][0].round(decimals=3):.2f}', transform=ax[1].transAxes)
# ax[1].set_xlabel(r'$log_{10}\ Predicted\ metabolome$')
# ax[1].set_ylabel(r'$log_{10}\ Empirical\ data$')
fig.supxlabel(r'$log_{10}\ Predicted\ metabolome$')
plt.tight_layout()

if figsave_flag:
    fig.savefig(fig_path+'/with-balance-prediction-comparison-reward-'+str(reward_arr[0])+'.png', dpi=300)
    print("Figure saved at "+fig_path)
    plt.close(fig)
else:
    plt.show()
# plt.scatter(np.log10(metabolome_pred+1e-7), np.log10(metabolome_measured+1e-7), c='k', s=4)

    
# %%
df_summary = pd.concat([df_con_links, df_pro_links])

plot_networks(net_ori, df_summary, 'original-net', n_reps, fig_path, figsave_flag)
plot_networks(net_optim, df_summary, 'optimised-net', n_reps, fig_path, figsave_flag)

