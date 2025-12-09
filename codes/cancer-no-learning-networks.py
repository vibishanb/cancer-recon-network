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
    b2m = csr_matrix( (data,(row,col)), shape=(MAX_ID_celltypes, MAX_ID_metabolites)).toarray()#.todense()

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
# i_nonzero_metabolites = np.sort(i_nonzero_metabolites)

MAX_ID_celltypes = len(i_nonzero_celltypes)  # MAX_ID_celltypes is the maximum of ID labels for celltypes.
MAX_ID_metabolites = len(i_nonzero_metabolites)  # MAX_ID_metabolites is the maximum of ID labels for metabolites.

diet = met_baseline.mean(axis=1)

####### Generate random networks, one for each cell line
# for i in range(len(all_networks)):
all_random_networks = []
for i, net in enumerate(all_networks):
    bias_metabolome = ec_metabolome.iloc[:, i].values - diet.values
    rnet = generate_random_network(net, bias_metabolome)
    all_random_networks.append(rnet)

# %%
###### Random production levels and celltype abundance for one cell line
f = 0.55
in_degree_flag = False

net, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(all_random_networks[0])
ec_real = ec_metabolome.iloc[:, 0].values

rmse_arr = []
n_produced_arr = []
cf_arr = []
for i in range(1, 11):
    x = np.where(net.iloc[:, -1] > 0, 1, 0)

    max_links = MAX_ID_metabolites * MAX_ID_celltypes

    n_pro = i*10
    edges = np.append(np.zeros(MAX_ID_metabolites-n_pro), np.ones(n_pro))
    x[max_links:] = np.random.permutation(np.repeat(edges[np.newaxis, :], 2, axis=0).ravel())

    net_mod = net_from_x(x, MAX_ID_metabolites=MAX_ID_metabolites, MAX_ID_celltypes=MAX_ID_celltypes)
    for cf in np.linspace(0.1, 0.9, 10):
        ct_freq = np.array([cf, 1-cf])
        m2b, b2m, ec_pred = calculate_metabolome_from_net(f, ct_freq, diet, net_mod, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

        i_nonzero = np.where(ec_pred * ec_real, True, False)
        i_filt = np.where((b2m.sum(0) > 0), True, False)
        i_final = i_nonzero * i_filt

        diff = np.log10(ec_pred[i_final]) - np.log10(ec_real[i_final]) # / np.log10(ec_real[i_nonzero])
        pred_error = np.sqrt(np.mean(diff**2)) #np.sqrt(np.dot(pred_error, pred_error.T)) #np.sqrt(np.sum(pred_error**2))

        rmse_arr.append(pred_error)
        cf_arr.append(cf)
        n_produced_arr.append(n_pro)

# %%
df = pd.DataFrame({'CT_Freq': cf_arr,
                   'N_produced': n_produced_arr,
                   'RMSE': rmse_arr})

sns.lineplot(data=df, x='N_produced', y='RMSE', hue='CT_Freq', palette='crest')

# %%
###### Randomly chosen subsets of "interconversion"; celltype 1 consumes what celltype 2 produces, and vice versa
f = 0.55
in_degree_flag = False

net, i_nonzero_celltypes, i_nonzero_metabolites, MAX_ID_celltypes, MAX_ID_metabolites = get_network(all_random_networks[0])
ec_real = ec_metabolome.iloc[:, 0].values

rmse_arr = []
n_produced_arr = []
cf_arr = []
rep_arr = []
ec_pred_arr = [[]]
index_arr = [[]]
n_reps = 10

# count = 1
for i in np.repeat(np.arange(1, 6), n_reps):
    max_links = MAX_ID_metabolites * MAX_ID_celltypes
    n_pro = i*10
    # count = i
    ### Edges for celltype 1
    selected_edges_random = np.random.choice(np.arange(MAX_ID_metabolites), size=n_pro*2, replace=False)
    consumption_edges = selected_edges_random[:n_pro]
    production_edges = selected_edges_random[n_pro:]

    ## Assigning celltype 1 edges
    net_consumption = net.iloc[:max_links, :]
    nc1 = net_consumption[net_consumption['celltypes']==0]
    nc2 = net_consumption[net_consumption['celltypes']==1]
    nc1.iloc[consumption_edges, -1] = np.repeat(2, n_pro)
    nc2.iloc[consumption_edges, -1] = np.repeat(0, n_pro)
    nc1.iloc[~consumption_edges, -1] = np.repeat(0, n_pro)
    nc2.iloc[~consumption_edges, -1] = np.repeat(2, n_pro)
    net_consumption = pd.concat([nc1, nc2])

    net_production = net.iloc[max_links:, :]
    np1 = net_production[net_production['celltypes']==0]
    np2 = net_production[net_production['celltypes']==1]
    np1.iloc[production_edges, -1] = np.repeat(3, n_pro)
    np2.iloc[production_edges, -1] = np.repeat(0, n_pro)
    np1.iloc[~production_edges, -1] = np.repeat(0, n_pro)
    np2.iloc[~production_edges, -1] = np.repeat(3, n_pro)
    net_production = pd.concat([np1, np2])

    net_mod = pd.concat([net_consumption, net_production])
    x = np.where(net_mod.iloc[:, -1] > 0, 1, 0)
    # edges = np.append(np.zeros(MAX_ID_metabolites-n_pro), np.ones(n_pro))
    # x[max_links:] = np.random.permutation(np.repeat(edges[np.newaxis, :], 2, axis=0).ravel())
    net_mod = net_from_x(x, MAX_ID_metabolites=MAX_ID_metabolites, MAX_ID_celltypes=MAX_ID_celltypes)
    
    for cf in np.linspace(0.1, 0.9, 10):
        ct_freq = np.array([cf, 1-cf])
        m2b, b2m, ec_pred = calculate_metabolome_from_net(f, ct_freq, diet, net_mod, in_degree_flag, MAX_ID_metabolites, MAX_ID_celltypes)

        i_nonzero = np.where(ec_pred * ec_real, True, False)
        i_filt = np.where((b2m.sum(0) > 0), True, False)
        i_final = i_nonzero * i_filt

        diff = np.log10(ec_pred[i_final]) - np.log10(ec_real[i_final]) # / np.log10(ec_real[i_nonzero])
        pred_error = np.sqrt(np.mean(diff**2)) #np.sqrt(np.dot(pred_error, pred_error.T)) #np.sqrt(np.sum(pred_error**2))

        rmse_arr.append(pred_error)
        cf_arr.append(cf)
        n_produced_arr.append(n_pro)
        ec_pred_arr.append(ec_pred)
        index_arr.append(i_final)
    # rep_arr.append(np.repeat(count, n_reps))
    # count += 1s

ec_pred_arr = np.array(ec_pred_arr[1:])
# %%
df = pd.DataFrame({'CT_Freq': cf_arr,
                   'N_produced': n_produced_arr,
                #    'Replicate': np.array(rep_arr),
                   'RMSE': rmse_arr})

sns.pointplot(data=df, x='CT_Freq', y='RMSE', hue='N_produced', palette='crest',
             markers=True, estimator='mean', errorbar='sd')

# %%
