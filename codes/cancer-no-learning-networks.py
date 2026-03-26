"""
<!---------------------------
Name: cancer-recon-network
File: cancer-no-learning-networks
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
from numpy import matlib
from tqdm import tqdm
# %%
############ Figure size settings
SMALL_SIZE = 14
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

def calculate_metabolome_from_net(f, ct_freq, diet, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
    '''
    This is a function used to create sparse matrices made of metabolites and celltypes 
    where metabolite consumption and production is considered. The matrices created are "m2b" and "b2m":
    (1) m2b is a matrix determines nutrient uptake by celltypes, and
    (2) b2m is a matrix determines the nutrient secretion.
    Both matrices have rows representing cancer celltypes and columns representing metabolites.
    Two matrices are created based on (1) the metabolite consumption and production network which is 
    encode in "net" as a dataframe, and (2) the hypothesised relative cell type frequencies "ct_freq". Those metabolites that are not taken up by any celltype are inherited as is from the diet supplied, while uptake of metabolites is assumed to be complete i.e., even if only a single celltype can take up a metabolite, it consumes all of it from the environment.
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
    b2m = csr_matrix((data,(row,col)), shape=(MAX_ID_celltypes, MAX_ID_metabolites)).toarray()#.todense()

    if in_degree_flag:
        m2b = m2b/in_degree
    
    ##### Intake matrix calculation
    cellnum = np.matlib.repmat(ct_freq[:, np.newaxis], 1, MAX_ID_metabolites) 
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

def calc_pred_error(ct_freq, net, f, diet, ec_real, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes):
    '''
    pred_error is a function used to compute the logarithmic error between experimentally measured
    metagenome and predicted metagenome computed from the model for a certain nutrient intake. It relies on 
    (1) diet: the nutrient intake, (2) i_intake: IDs of the nutrient intake, (3) m2b_total: a conversion matrix 
    from the nutrient intake to the total biomass, and (4) ct_freq: hypothesised relative cell type frequenices. The 
    first three is used to compute the net gain in the intracellular metabolome predicted by the model "ic_pred" and compare it with the 
    experimentally measured net gain in intracellular metabolome "ic_real".
    '''

    m2b, b2m, ec_pred = calculate_metabolome_from_net(f, ct_freq, diet, net, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)
    
    i_nonzero = np.where((ec_pred * ec_real) > 0)[0]
    diff = np.log10(ec_pred[i_nonzero]) - np.log10(ec_real[i_nonzero]) # / np.log10(ec_real[i_nonzero])
    pred_error = np.sqrt(np.mean(diff**2)) #np.sqrt(np.dot(pred_error, pred_error.T)) #np.sqrt(np.sum(pred_error**2))
    
    return pred_error


def generate_random_network(net, bias_metabolome):
    rnet = net.copy()
    # rnet.iloc[:, -1] = 0
    # n_unused = (rnet.iloc[:, -1] == 0).sum() # Number of unused metabolites
    # n_edges = rnet.shape[0]
    i_consumption_edges = np.where(bias_metabolome > 0)[0]
    i_production_edges = np.where(bias_metabolome < 0)[0]

    con_edges = np.random.choice([0, 2], len(i_consumption_edges), replace=True)
    pro_edges = np.random.choice([0, 3], len(i_production_edges), replace=True)
    rnet.iloc[i_consumption_edges, -1] = con_edges.copy()
    rnet.iloc[i_production_edges, -1] = pro_edges.copy()

    assigned_edges = rnet.iloc[:len(bias_metabolome), -1].values
    for i in range(1, MAX_ID_celltypes):
        shuffled_edges = assigned_edges.copy()
        shuffled_edges[i_production_edges] = np.random.permutation(shuffled_edges[i_production_edges])
        assigned_edges = np.append(assigned_edges, shuffled_edges)

    rnet.iloc[:, -1] = assigned_edges.copy() #shuffled_edges.copy()

    return rnet

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
    max_links = MAX_ID_celltypes * MAX_ID_metabolites 
    prior_prob += 1/max_links # All links have some constant probability to be chosen at random, independent of the bias in the metabolome prediction
    prior_prob /= prior_prob.sum(0) # Normalise to [0, 1]
    return prior_prob

# %%
###### Algorithm to start with a single celltype production and consumption profile, and split it into two celltypes obeying production-consumption balance with and without production partitioning
f = 0.5
in_degree_flag = False
home_dir = os.getcwd()
all_networks, i_intake, names = pd.read_pickle(home_dir + '/1-cells-cancer_network.pickle')

home_dir = os.getcwd()
all_networks, i_intake, names = pd.read_pickle(home_dir + '/1-cells-cancer_network.pickle')
# i_selfish = 0

# pickle_in = open("data.pickle","rb")
celltype_ID, celltypefreq, ec_metabolome_ID, ec_metabolome, met_baseline, core_mean = pd.read_pickle(home_dir + '/1-cells-data.pickle')
cell_line_names = ec_metabolome.columns.to_numpy()

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

slope_df, sublinear_cell_lines = pd.read_pickle(home_dir + '/cancer_power_law_stats.pickle')
nets_temp = all_networks.copy()

### Filtering those networks for cell lines with power law slopes
i_sublinear = np.where(np.isin(ec_metabolome.columns.values, sublinear_cell_lines))[0]
all_networks = []
for i in i_sublinear:
    all_networks.append(nets_temp[i])

# net, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(all_networks[0])

ec_metabolome = ec_metabolome.iloc[:, i_sublinear]
diet = met_baseline.mean(axis=1)
# i_nonzero_metabolites = np.sort(i_nonzero_metabolites)

cellnum_init_all = cellnum_init_all[i_sublinear]
cellnum_final_all = cellnum_final_all[i_sublinear]

net_state = 'no-learn-balanced-net/'
for n_ct in np.arange(6, 7):

    n_rand = 100
    n_reps = 150
    # npro_arr = np.repeat(np.array([50])[np.newaxis, :], n_reps*2, axis=1).ravel()
    # nct_arr = np.repeat(np.array([[1]*n_reps, [2]*n_reps]).ravel()[np.newaxis, :], 1, axis=0).ravel()
    for j in tqdm(range(len(i_sublinear[:1])), desc='Cell line: '):
        rmse_arr_1ct, rmse_arr_nct = [], []
        balance_flag_arr = []
        balance_arr_1ct, balance_arr_nct = [], [[]]
        n_produced_arr_1ct, n_produced_arr_nct = [], []
        cf_arr = [[]]
        rep_arr = []
        ec_pred_arr_1ct, ec_pred_arr_nct = [[]], [[]]
        index_arr_1ct, index_arr_nct = [[]], [[]]
        production_ct_arr_1ct, production_ct_arr_nct = [[]], [[]]
        consumption_ct_arr_1ct, consumption_ct_arr_nct = [[]], [[]]
        nct_arr = []
        final_nets_arr = []
        balanced_networks_list = []

        ec_real = ec_metabolome.iloc[:, j].values
        low_mets = np.where(ec_real <= np.median(ec_real))[0]
        high_mets = np.where(ec_real > np.median(ec_real))[0]

        for i in tqdm(range(n_rand), desc='Replicate: ', leave=False):

            net, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(all_networks[j])
            max_links = MAX_ID_celltypes * MAX_ID_metabolites
            npro = MAX_ID_metabolites
            
            ### Random starting network with one celltype
            all_edges = np.arange(MAX_ID_metabolites)
            net_consumption = net.iloc[:max_links, :]
            net_production = net.iloc[max_links:, :]

            chosen_edges = np.random.choice(all_edges, size=npro, replace=False) # Select a random subset of 'npro' mets for the one celltype model to produce and consume
            # net_consumption.iloc[:, -1] = np.zeros(max_links)
            net_consumption.loc[:, 'edgeType'] = np.where(np.isin(np.arange(MAX_ID_metabolites), chosen_edges), 2, 0)
            net_production.loc[:, 'edgeType'] = np.where(np.isin(np.arange(MAX_ID_metabolites), chosen_edges), 3, 0)
            net_1ct = pd.concat([net_consumption, net_production])
            #### For max celltypes = 1, no model fitting, final cell number is taken directly from cell number at confluency
            ct_final = cellnum_final_all[j]*celltypefreq.values

            ### Calculate predicted metabolome for the single celltype
            m2b, b2m, ec_pred = calculate_metabolome_from_net(f, ct_final, diet, net_1ct, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes=1)
            cf_final = ct_final[0]/ct_final.sum()

            i_nonzero = np.where(ec_pred * ec_real, True, False)
            i_filt = np.where((b2m.sum(0) > 0), True, False)
            i_final = i_nonzero * i_filt

            diff = np.log10(ec_pred[i_final]) - np.log10(ec_real[i_final]) # / np.log10(ec_real[i_nonzero])
            pred_error = np.sqrt(np.mean(diff**2)) #np.sqrt(np.dot(pred_error, pred_error.T)) #np.sqrt(np.sum(pred_error**2))

            balance = ec_pred[chosen_edges].sum()/diet.values[chosen_edges].sum()
            
            p_arr = np.array([i*j for i, j in zip(b2m, [1, 2])]).sum(0)
            c_arr = np.array([i*j for i, j in zip(m2b, [1, 2])]).sum(0)
            rmse_arr_1ct.append(pred_error)
            n_produced_arr_1ct.append(npro)
            ec_pred_arr_1ct.append(ec_pred)
            index_arr_1ct.append(i_final)
            production_ct_arr_1ct.append(p_arr)
            consumption_ct_arr_1ct.append(c_arr)
            balance_arr_1ct.append(balance)
            
            ### Splitting the above one celltype into a network with n_ct celltypes
            # n_ct = 4
            max_links = MAX_ID_metabolites*n_ct

            quantiles = np.quantile(ec_real, np.linspace(0, 1, n_ct+1)[1:-1])
            prod_celltypes = np.zeros_like(ec_real)
            for i in range(n_ct-1):
                prod_celltypes = np.where(ec_real <= quantiles[i], prod_celltypes, prod_celltypes+1)
            prod_celltypes = np.int64(prod_celltypes)
            # net_temp = net.copy()
            # net_temp.iloc[max_links:, 1] = np.ones(max_links)
            # net_nct = pd.concat([net_temp, net_temp])

            net_split = pd.concat([net]*n_ct)
            net_split.loc[:, 'celltypes'] = np.concatenate([np.repeat(np.arange(n_ct), MAX_ID_metabolites)]*2)
            net_split.loc[:, 'edgeType'] = 0
            net_consumption = net_split.iloc[:max_links, :]
            net_production = net_split.iloc[max_links:, :]

            ### Production edges with partition-no metabolites are left unproduced
            for i in range(n_ct):
                i_prod = np.where(prod_celltypes == i, True, False)
                ct_index = np.where(net_production.loc[:, 'celltypes']==i)[0]

                pro_edges = net_production.loc[:, 'edgeType'].values
                pro_edges[ct_index[i_prod]] = 3
                pro_edges[ct_index[~i_prod]] = 0
                
                net_production.loc[:, 'edgeType'] = pro_edges.copy()
                # net_production.iloc[ct_index[i_prod], -1] = 3
                # net_production.iloc[ct_index[~i_prod], -1] = 0

            ### Within the interconversion regime, consumption is random
            # num_con = np.array([np.random.randint(1, npro-1) for i in range(n_ct-1)])
            # num_con = np.append(num_con, npro-num_con.sum())
            rand = np.random.exponential(5, n_ct) # np.random.uniform(0, 1, n_ct)
            rand = rand/rand.sum()
            num_con = (rand*MAX_ID_metabolites).astype(int)
            delta = num_con.sum() - MAX_ID_metabolites
            i_rand = np.random.choice(np.arange(n_ct), 1)
            num_con[i_rand] -= delta
            num_con = np.sort(num_con)
            con_celltypes = np.random.permutation(np.repeat(np.arange(n_ct), num_con))
            
            for i in range(n_ct):
                i_con = np.where(con_celltypes == i, True, False)  
                ct_index = np.where(net_consumption.loc[:, 'celltypes']==i)[0]

                con_edges = net_consumption.loc[:, 'edgeType'].values
                con_edges[ct_index[i_con]] = 2
                con_edges[ct_index[~i_con]] = 0
                
                net_consumption.loc[:, 'edgeType'] = con_edges.copy()

                # net_consumption.iloc[ct_index[i_con], -1] = 2
                # net_consumption.iloc[ct_index[~i_con], -1] = 0
            net_nct = pd.concat([net_consumption, net_production])

            cf = np.round(np.random.uniform(0, 1, n_ct), decimals=2)
            cf = cf/cf.sum()
            ct0_nct = cf*cellnum_init_all[j]

            ### Learn the best celltype frequency
            ct_final = np.zeros_like(ct0_nct) # Final cell number, either fitted or taken depending on number of cell types
            my_args = (net_nct, f, diet, ec_real, in_degree_flag, MAX_ID_metabolites, n_ct)
            bnds = ((0, cellnum_final_all[j]), ) * len(ct0_nct)
            constraint = {'type': 'eq', 'fun': lambda ct: ct.sum() - cellnum_final_all[j]}
            res = minimize(calc_pred_error, ct0_nct, args=my_args, method='SLSQP', bounds=bnds, options={'disp': False, 'maxiter': 1000}, tol=1e-3, constraints=constraint)
            ct_final = res.x
            
            ### Calculate predicted metabolome using the fitted celltype frequencies
            m2b, b2m, ec_pred = calculate_metabolome_from_net(f, ct_final, diet, net_nct, in_degree_flag, MAX_ID_metabolites, n_ct)

            #### Production-consumption balance for the two celltype network
            ## Balance calculation
            balance = np.zeros(n_ct)
            for i in range(n_ct):
                net_temp = net_nct[net_nct['celltypes']==i]
                i_consumed = np.where(net_temp.iloc[:MAX_ID_metabolites, -1]==2)[0]
                i_produced = np.where(net_temp.iloc[MAX_ID_metabolites:, -1]==3)[0]
                flux_consumption = diet.values[i_consumed].sum()
                flux_production = ec_real[i_produced].sum()
                balance[i] = flux_production/flux_consumption

            # If any part of the balance is violated for either celltype, resample links till the balance is satisfied
            count = 0
            while (count <= n_reps) * ((balance < 0.05)+(balance > 2)).any():
                # print("Sampled links rejected for balance values " + str(Xpro_ct1) + " and " + str(Xpro_ct2))
                # con_celltypes = np.random.choice(np.arange(n_ct), len(prod_celltypes))
                rand = np.random.exponential(5, n_ct) # np.random.uniform(0, 1, n_ct)
                rand = rand/rand.sum()
                num_con = (rand*MAX_ID_metabolites).astype(int)
                delta = num_con.sum() - MAX_ID_metabolites
                i_rand = np.random.choice(np.arange(n_ct), 1)
                num_con[i_rand] -= delta
                num_con = np.sort(num_con)
                con_celltypes = np.random.permutation(np.repeat(np.arange(n_ct), num_con))
                for i in range(n_ct):
                    i_con = np.where(con_celltypes == i, True, False)  
                    ct_index = np.where(net_consumption.loc[:, 'celltypes']==i)[0]
                    con_edges = net_consumption.loc[:, 'edgeType'].values
                    con_edges[ct_index[i_con]] = 2
                    con_edges[ct_index[~i_con]] = 0

                    net_consumption.loc[:, 'edgeType'] = con_edges.copy()
                    # net_consumption.iloc[ct_index[i_con], -1] = 2
                    # net_consumption.iloc[ct_index[~i_con], -1] = 0

                net_nct = pd.concat([net_consumption, net_production])

                ### Learn the best celltype frequency
                ct_final = np.zeros_like(ct0_nct)
                my_args = (net_nct, f, diet, ec_real, in_degree_flag, MAX_ID_metabolites, n_ct)
                bnds = ((0, cellnum_final_all[j]), ) * len(ct0_nct)
                constraint = {'type': 'eq', 'fun': lambda ct: ct.sum() - cellnum_final_all[j]}
                res = minimize(calc_pred_error, ct0_nct, args=my_args, method='SLSQP', bounds=bnds, options={'disp': False, 'maxiter': 1000}, tol=1e-3, constraints=constraint)
                ct_final = res.x #res.x.max()/cellnum_max
                
                ### Calculate predicted metabolome using the fitted celltype frequencies
                m2b, b2m, ec_pred = calculate_metabolome_from_net(f, ct_final, diet, net_nct, in_degree_flag, MAX_ID_metabolites, n_ct)

                #### Recalculate production-consumption balance
                balance = np.zeros(n_ct)
                for i in range(n_ct):
                    net_temp = net_nct[net_nct['celltypes']==i]
                    i_consumed = np.where(net_temp.iloc[:MAX_ID_metabolites, -1]==2)[0]
                    i_produced = np.where(net_temp.iloc[MAX_ID_metabolites:, -1]==3)[0]
                    flux_consumption = diet.values[i_consumed].sum()
                    flux_production = ec_real[i_produced].sum()
                    balance[i] = flux_production/flux_consumption
                count += 1

            if ((balance < 0.05)+(balance > 2)).any():
                balance_flag = False
                # print("Balance failed")
            else:
                balance_flag = True
                # print("Sampled links accepted for balance values " + str(balance.round(decimals=3).tolist()) + " and " + str(num_con.tolist()) + " mets consumed")
            
            balance_flag_arr.append(balance_flag)
            balance_arr_nct.append(balance)

            ### Learn the best celltype frequency
            ct_final = np.zeros_like(ct0_nct) # Final cell number, either fitted or taken depending on number of cell types
            my_args = (net_nct, f, diet, ec_real, in_degree_flag, MAX_ID_metabolites, n_ct)
            bnds = ((0, cellnum_final_all[j]), ) * len(ct0_nct)
            constraint = {'type': 'eq', 'fun': lambda ct: ct.sum() - cellnum_final_all[j]}
            res = minimize(calc_pred_error, ct0_nct, args=my_args, method='SLSQP', bounds=bnds, options={'disp': False, 'maxiter': 1000}, tol=1e-3, constraints=constraint)
            ct_final = res.x #res.x.max()/cellnum_max
            
            ### Calculate predicted metabolome using the fitted celltype frequencies
            m2b, b2m, ec_pred = calculate_metabolome_from_net(f, ct_final, diet, net_nct, in_degree_flag, MAX_ID_metabolites, n_ct)
            cf_final = ct_final/ct_final.sum()

            i_nonzero = np.where(ec_pred * ec_real, True, False)
            i_filt = np.where((b2m.sum(0) > 0), True, False)
            i_final = i_nonzero * i_filt

            diff = np.log10(ec_pred[i_final]) - np.log10(ec_real[i_final]) # / np.log10(ec_real[i_nonzero])
            pred_error = np.sqrt(np.mean(diff**2)) #np.sqrt(np.dot(pred_error, pred_error.T)) #np.sqrt(np.sum(pred_error**2))
            
            p_arr = np.array([i*j for i, j in zip(b2m, np.arange(1, n_ct+1))]).sum(0)
            c_arr = np.array([i*j for i, j in zip(m2b, np.arange(1, n_ct+1))]).sum(0)
            rmse_arr_nct.append(pred_error)
            cf_arr.append(cf_final.round(decimals=2))
            n_produced_arr_nct.append(npro)
            ec_pred_arr_nct.append(ec_pred)
            index_arr_nct.append(i_final)
            production_ct_arr_nct.append(p_arr)
            consumption_ct_arr_nct.append(c_arr)
            balanced_networks_list.append(net_nct)

        ec_pred_arr_1ct, ec_pred_arr_nct = np.array(ec_pred_arr_1ct[1:]), np.array(ec_pred_arr_nct[1:])
        index_arr_1ct, index_arr_nct = np.array(index_arr_1ct[1:]), np.array(index_arr_nct[1:])
        production_ct_arr_1ct, production_ct_arr_nct = np.array(production_ct_arr_1ct[1:]), np.array(production_ct_arr_nct[1:])
        consumption_ct_arr_1ct, consumption_ct_arr_nct = np.array(consumption_ct_arr_1ct[1:]), np.array(consumption_ct_arr_nct[1:])
        rmse_arr_1ct, rmse_arr_nct = np.array(rmse_arr_1ct), np.array(rmse_arr_nct)
        cf_arr = np.array(cf_arr[1:])
        n_produced_arr_1ct, n_produced_arr_nct = np.array(n_produced_arr_1ct), np.array(n_produced_arr_nct)
        balance_flag_arr = np.array(balance_flag_arr)
        balance_arr_1ct, balance_arr_nct = np.array(balance_arr_1ct), np.array(balance_arr_nct[1:])

        pickle_path = '../raw-output/'+str(n_ct)+'-celltypes/'+net_state+sublinear_cell_lines[j]
        try:
            os.makedirs(pickle_path)
        except:
            pass
        pickle_out = open(pickle_path+"/balanced-networks.pickle", "wb")
        #pickle.dump([net, i_selfish, i_intake, names], pickle_out)
        pickle.dump([balance_arr_1ct, balance_arr_nct, balance_flag_arr, balanced_networks_list,
                    rmse_arr_1ct, rmse_arr_nct,
                    ec_pred_arr_1ct, ec_pred_arr_nct,
                    index_arr_1ct, index_arr_nct,
                    production_ct_arr_1ct, production_ct_arr_nct], pickle_out, protocol=2)
        pickle_out.close()

# %%
net_state = 'no-learn-balanced-net/'
figsave_flag = 0
fig_path = '../figures/'+str(n_ct)+'-celltypes/'+net_state+ec_metabolome.columns[0]
try:
    os.makedirs(fig_path)
except:
    pass

df = pd.DataFrame({'Replicate': np.concatenate([np.arange(1, n_rand+1), np.arange(1, n_rand+1)]),
                   'N_CT': np.array([np.ones_like(rmse_arr_1ct), np.zeros_like(rmse_arr_1ct)+n_ct]).ravel().astype(int),
                   'Balance': np.concatenate([balance_flag_arr, balance_flag_arr]),
                   'X_pro': np.concatenate([balance_arr_1ct, balance_arr_nct[:, 1]]),
                   'RMSE': np.concatenate([rmse_arr_1ct, rmse_arr_nct])})

g = sns.lineplot(data=df, x='N_CT', y='RMSE',
                units='Replicate', hue='Balance',
                palette='coolwarm_r', hue_norm=(0, 1),
                dashes=False, estimator=None, zorder=1, legend=True)
g = sns.scatterplot(data=df, x='N_CT', y='RMSE',
                    hue='Balance', palette='coolwarm_r', hue_norm=(0, 1),
                    edgecolor='face', alpha=0.8, zorder=2, legend=False)
g.set_xlabel(r'$N_{CT}$')
g.set(xlim=(0.5, n_ct+0.5), xticks=[1, n_ct])

if figsave_flag:
    g.figure.savefig(fig_path+'/balance-pairwise-comparison-npro-'+str(npro)+'.png', dpi=300)
    plt.close(g.figure)
else:
    plt.show()

h = sns.boxplot(data=df, x='Balance', y='RMSE',
                hue='N_CT', palette='crest')
h.set_xlabel('Balance state')
h.set(xlim=(-0.75, 1.75), xticks=[0, 1], xticklabels=['False', 'True'])
h.get_legend().set_title(r'$N_{CT}$')
if figsave_flag:
    h.figure.savefig(fig_path+'/balance-comparison-boxplot-npro-'+str(npro)+'.png', dpi=300)
    plt.close(h.figure)
else:
    plt.show()

# prod_df = pd.DataFrame({'Xpro_CT1': balance_arr_nct[:, 0],
#                         'Xpro_CT2': balance_arr_nct[:, 1],
#                         'RMSE': rmse_arr_nct})
prod_df = pd.DataFrame(balance_arr_nct, columns=np.arange(1, n_ct+1)).join(pd.Series(rmse_arr_nct, name='RMSE')).melt(id_vars='RMSE', value_name='Balance', var_name='N_CT')

q = sns.scatterplot(data=prod_df, x='Balance', y='RMSE',
                    hue = 'N_CT', palette='flare',#sizes=(50, 200),
            # hue='N_produced', palette='crest',
            edgecolor= 'face', alpha = 0.8, s=75)
q.set_xscale('log')
q.axvspan(xmin=0.1, xmax=1, alpha=0.2, color='tab:green')
# q.set_yscale('log')
q.set_xlabel(r'$\chi_{production}$')
q.set_ylabel('RMSE')
# q.axhline(y=1, linestyle='dashed', c='tab:green')
# # q.text(0.9, 0.38, r'$y=1$', c='tab:green', transform=g.transAxes)
# q.axvline(x=1, linestyle='dashed', c='tab:green')
# # q.text(0.68, 0.87, r'$x=1$', c='tab:green', rotation='vertical', transform=g.transAxes)

# prod_df = pd.DataFrame({'Celltype': np.array([np.ones_like(rmse_arr_nct), np.ones_like(rmse_arr_nct)+1]).ravel().astype(int),
#                    'X_pro': np.concatenate([balance_arr_nct[:, 0], balance_arr_nct[:, 1]]),
#                    'RMSE': np.concatenate([rmse_arr_nct, rmse_arr_nct])})


# q = sns.lineplot(data=prod_df, x='X_pro', y='RMSE',
#                     hue='Celltype', palette=['b', 'g'], lw=3)
# q.set_xscale('log')
# q.set_xlabel(r'$\chi_{production}$', fontsize=17)
# q.set_ylabel(r'RMSE')


if figsave_flag:
    q.figure.savefig(fig_path+'/balance-and-rmse-lineplot-npro-'+str(npro)+'.png', dpi=300)
    plt.close(q.figure)
else:
    plt.show()

# %%
figsave_flag = 0

f, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8.5, 3.5))

rmse_diff = rmse_arr_1ct - rmse_arr_nct
i_min_diff = np.where(rmse_diff == rmse_diff.min(), True, False)
i_max_diff = np.where(rmse_diff == rmse_diff.max(), True, False)

#### Minimum difference
colors = np.where(production_ct_arr_1ct[i_min_diff]==1, 'b', 
                  np.where(production_ct_arr_1ct[i_min_diff]==2, 'g',
                           np.where(production_ct_arr_1ct[i_min_diff]==3, 'm',
                                    np.where(production_ct_arr_1ct[i_min_diff]==4, 'saddlebrown',
                                             np.where(production_ct_arr_1ct[i_min_diff]==5, 'cyan', 'teal')))))
labels = np.where(production_ct_arr_1ct[i_min_diff]==1, 'CT1', 
                  np.where(production_ct_arr_1ct[i_min_diff]==2, 'CT2',
                           np.where(production_ct_arr_1ct[i_min_diff]==3, 'CT3',
                                    np.where(production_ct_arr_1ct[i_min_diff]==4, 'CT4',
                                             np.where(production_ct_arr_1ct[i_min_diff]==5, 'CT5', 'CT6')))))

ax[0].scatter(np.log10(ec_pred_arr_1ct[i_min_diff][index_arr_1ct[i_min_diff]]), 
                np.log10(ec_real[index_arr_1ct[i_min_diff][0]]),
            c=colors[index_arr_1ct[i_min_diff]],#'tab:blue',
            alpha=0.6, edgecolor='face')
ax[0].axline((-2, -2), (3, 3), c='k', ls='--')
ax[0].set_title(r'$N_{CT} = 1$')
ax[0].text(0.05, 0.9, f'RMSE = {rmse_arr_1ct[i_min_diff][0]:.2f}', transform=ax[0].transAxes)
# ax[0].text(0.05, 0.8, str(balance_arr_1ct[i_min_diff][0].round(decimals=2)), transform=ax[0].transAxes)

colors = np.where(production_ct_arr_nct[i_min_diff]==1, 'b', 
                  np.where(production_ct_arr_nct[i_min_diff]==2, 'g',
                           np.where(production_ct_arr_nct[i_min_diff]==3, 'm',
                                    np.where(production_ct_arr_nct[i_min_diff]==4, 'saddlebrown',
                                             np.where(production_ct_arr_nct[i_min_diff]==5, 'cyan', 'black')))))
labels = np.where(production_ct_arr_nct[i_min_diff]==1, 'CT1', 
                  np.where(production_ct_arr_nct[i_min_diff]==2, 'CT2',
                           np.where(production_ct_arr_nct[i_min_diff]==3, 'CT3',
                                    np.where(production_ct_arr_nct[i_min_diff]==4, 'CT4',
                                             np.where(production_ct_arr_nct[i_min_diff]==5, 'CT5', 'CT6')))))

# con_index_ct1 = np.where(consumption_ct_arr_nct[i_min_diff]==0, True, False)[0]*index_arr_nct[i_min_diff]
# con_index_ct2 = np.where(consumption_ct_arr_nct[i_min_diff]==1, True, False)[0]*index_arr_nct[i_min_diff]
ax[1].scatter(np.log10(ec_pred_arr_nct[i_min_diff][index_arr_nct[i_min_diff]]), 
                np.log10(ec_real[index_arr_nct[i_min_diff][0]]),
            c=colors[index_arr_nct[i_min_diff]], marker='o',
            alpha=0.6, edgecolor='face')

ax[1].axline((-2, -2), (3, 3), c='k', ls='--')
ax[1].set_title(r'Worst of $N_{CT} = $'+str(n_ct))
ax[1].text(0.5, 0.9, f'RMSE = {rmse_arr_nct[i_min_diff][0]:.2f}', transform=ax[1].transAxes)
# ax[1].text(0.4, 0.8, str(balance_arr_nct[i_min_diff][0].round(decimals=2).tolist()), transform=ax[1].transAxes)
# ax[0, 1].text(1.02, 0.45, 'Worst', transform=ax[0, 1].transAxes, rotation=270)

#### Maximum difference
colors = np.where(production_ct_arr_nct[i_max_diff]==1, 'b', 
                  np.where(production_ct_arr_nct[i_max_diff]==2, 'g',
                           np.where(production_ct_arr_nct[i_max_diff]==3, 'm',
                                    np.where(production_ct_arr_nct[i_max_diff]==4, 'saddlebrown',
                                             np.where(production_ct_arr_nct[i_max_diff]==5, 'cyan', 'black')))))
labels = np.where(production_ct_arr_nct[i_max_diff]==1, 'CT1', 
                  np.where(production_ct_arr_nct[i_max_diff]==2, 'CT2',
                           np.where(production_ct_arr_nct[i_max_diff]==3, 'CT3',
                                    np.where(production_ct_arr_nct[i_max_diff]==4, 'CT4',
                                             np.where(production_ct_arr_nct[i_max_diff]==5, 'CT5', 'CT6')))))
# con_index_ct1 = np.where(consumption_ct_arr_nct[i_max_diff]==0, True, False)[0]*index_arr_nct[i_max_diff]
# con_index_ct2 = np.where(consumption_ct_arr_nct[i_max_diff]==1, True, False)[0]*index_arr_nct[i_max_diff]
ax[2].scatter(np.log10(ec_pred_arr_nct[i_max_diff][index_arr_nct[i_max_diff]]), 
                np.log10(ec_real[index_arr_nct[i_max_diff][0]]),
            c=colors[index_arr_nct[i_max_diff]], marker='o',
            alpha=0.6, edgecolor='face')

ax[2].axline((-2, -2), (3, 3), c='k', ls='--')
ax[2].text(0.6, 0.1, f'RMSE = {rmse_arr_nct[i_max_diff][0]:.2f}', transform=ax[2].transAxes)
# ax[2].text(0.4, 0.1, str(balance_arr_nct[i_max_diff][0].round(decimals=2).tolist()), transform=ax[2].transAxes)
ax[2].set_title(r'Best of $N_{CT} = $'+str(n_ct))
# ax[2].text(1.02, 0.45, 'Best', transform=ax[2].transAxes, rotation=270)

f.supxlabel(r'$log_{10}\ Predicted\ metabolome$', fontsize=15)
f.supylabel(r'$log_{10}\ Empirical\ metabolome$', fontsize=15)
# f.suptitle(r'Random sampled networks; $N_{produced} = $'+str(npro))
f.tight_layout()

if figsave_flag:
    f.figure.savefig(fig_path+'/metabolome-prediction-comparison-npro-'+str(npro)+'.png', dpi=300)
    plt.close(f.figure)
else:
    plt.show()

# %%
