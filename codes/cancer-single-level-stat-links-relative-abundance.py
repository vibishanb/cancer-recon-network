# # Trophic model for the gut microbiome
# The human gut microbial community is complex because of 3 reasons: (1) many microbial species ($\approx$ 570); (2) many metabolites involved ($\approx$ 244); and (3) many microbe-metabolite interactions/links (>4400). Of all types of microbe-metabolite interactions/links, the cross-feeding makes the system more complicated to interpret. Previously, a literature-curated interspecies network of the human gut microbiota, called [NJS16](https://www.nature.com/articles/ncomms15393) is reported. This is an extensive data resource composed of ∼570 microbial species and 3 human cell types metabolically interacting through >4,400 small-molecule transport and macromolecule degradation events.
# 
# Here, we devoted to building a Consumer-Resource model (CRM) with trophic levels posted in the paper "[an evidence for a multi-level trophic organization of the human gut microbiome](https://www.biorxiv.org/content/10.1101/603365v1.full-text)". The model is devoted to studying the complex microbe-metabolite network in human guts.
# 
# In concise, a trophic level is considered as one round of carbon processing and is composed of two processes: resource allocation to celltypes and the following resource/byproduct generation by celltypes. The simulation is stopped after several trophic levels because the residence time of resources in the human gut is assumed to be finite.

# %%
import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import sem

#%% Plot Tong's default setting
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
########### import the pickled file containing all processed data which are useful for simulations (the processing is
########### done in "Trophic_model_for_gut_data_processing.ipynb")
import pickle
# pickle_in = open("cancer_network.pickle","rb")
# net, i_intake, names = pickle.load(pickle_in)
net, i_intake, names = pd.read_pickle("cancer_network.pickle")
# i_selfish = 0

# pickle_in = open("data.pickle","rb")
# celltype_ID, celltypefreq, ec_metabolome_ID, ec_metabolome, ic_metabolome_ID, ic_metabolome = pickle.load(pickle_in)
celltype_ID, celltypefreq, ec_metabolome_ID, ec_metabolome, ic_metabolome_ID, ic_metabolome = pd.read_pickle("data.pickle")


# Create maps of celltypes and metabolites to their reduced matrix forms

# %%
i_nonzero_celltypes = net['celltypes_ID'].unique()
i_nonzero_celltypes = np.sort(i_nonzero_celltypes)
i_nonzero_celltypes = celltype_ID.values.copy()
i_nonzero_metabolites = net['metabolites_ID'].unique()
i_nonzero_metabolites = np.sort(i_nonzero_metabolites)


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
net = net_reduced.copy()
net_temp = net.copy()
net['edgeType'][net['edgeType']==5] = 2
net_temp['edgeType'][net_temp['edgeType']==5] = 3
net = pd.concat([net, net_temp]).drop_duplicates() #net.append(net_temp).drop_duplicates()
net_ori = net.copy()

celltype_ID_reduced = df_celltypes.reindex(celltype_ID).values.flatten()
celltype_ID = celltype_ID_reduced[~np.isnan(celltype_ID_reduced)].astype(int)

metabolome_ID_reduced = df_metabolites.reindex(ec_metabolome_ID).values.flatten()
metabolome_ID = metabolome_ID_reduced[~np.isnan(metabolome_ID_reduced)].astype(int)


# i_selfish_reduced = df_celltypes.reindex(i_selfish).values.flatten()
# i_selfish = i_selfish_reduced[~np.isnan(i_selfish_reduced)].astype(int)

i_intake_reduced = df_metabolites.loc[i_intake].values.flatten()
i_intake = i_intake_reduced[~np.isnan(i_intake_reduced)].astype(int)



# Run the simulation with reduced matrix forms for one individual

# %%
################################# Predict metabolome from individual's metagenome.
from numpy import array
from scipy.sparse import csr_matrix
import numpy.matlib
from scipy.optimize import minimize
from scipy.stats import pearsonr

MAX_ID_celltypes = len(i_nonzero_celltypes)  # MAX_ID_celltypes is the maximum of ID labels for celltypes.
MAX_ID_metabolites = len(i_nonzero_metabolites)  # MAX_ID_metabolites is the maximum of ID labels for metabolites.

def Ain_out(ct_hyp, x, net):
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
    in_degree = m2b.sum()

    valid_index = np.where((net['edgeType']==3) | (net['edgeType']==5))[0]
    row = net['metabolites'].iloc[valid_index]
    col = net['celltypes'].iloc[valid_index]
    data = np.ones((len(valid_index),))
    b2m = csr_matrix( (data,(row,col)), shape=(MAX_ID_metabolites, MAX_ID_celltypes)).toarray()#.todense()

    ########## Normalize the b2m by out_degree
    out_degree = b2m.sum(0).copy()
    out_degree[out_degree==0]=1e6
    b2m = (b2m / out_degree)
    # b2m = np.array([i * x.to_numpy(dtype=float) for i in b2m.T], dtype=float).T # Adding known external supply of metabolites to the uptake matrix

    ########## Normalize the m2b by proportion of microbial abundance in each individual
    # ct_hyp_repmat = numpy.matlib.repmat(ct_hyp[np.newaxis,:], MAX_ID_metabolites, 1)
    # m2b = m2b * ct_hyp_repmat
    # Also multiply by relative nutrient concentrations
    # in_degree = m2b.sum(1)
    # in_degree[in_degree==0]=1e6
    # m2b = m2b / numpy.matlib.repmat(in_degree[:,np.newaxis], 1, MAX_ID_celltypes)
    ct_hyp_repmat = numpy.matlib.repmat(np.array(celltypefreq), MAX_ID_metabolites, 1)
    m2b = m2b * ct_hyp_repmat # Uptake is proportional to cell type relative abundance
    m2b = np.asarray([i*j for i, j in zip(m2b, x.to_numpy(dtype=float))]) # Relative amount of nutrient taken up
    m2b = m2b/in_degree
    
    m2b = np.float32(m2b)
    b2m = np.float32(b2m)
    return [m2b, b2m]

def m2b_multiple_levels(f, m2b, b2m, numLevels_max):
    '''
    m2b_multiple_levels is a function used to generate matrices involving the calculation of metabolite 
    byproducts and microbial biomass after several trophic levels/layers. Those matrices are:
    (1) m2m_layer is a conversion matrix from the nutrient intake to the metabolite byproducts at a trophic
    level or layer.
    (2) m2m_total is a conversion matrix from the nutrient intake to a summation of metabolite byproducts at
    all trophic levels or layers.
    (3) m2b_total is a conversion matrix from the nutrient intake to a summation of all microbial/bacterial 
    biomass gain at all trophic levels or layers.
    Those matrices are computed based on (1) metabolite consumption matrix "m2b", (2) metabolite byproduct
    generation matrix "b2m", (3) byproduct/leakage fraction "f", and (4) number of trophic levels/layers in the 
    simulation "numLevels_max".
    '''
    m2m_layer = np.zeros((MAX_ID_metabolites, MAX_ID_metabolites, numLevels_max));  
    #m2b_total = np.zeros((MAX_ID_metabolites, MAX_ID_microbes));  
    m2b_total = np.zeros((MAX_ID_metabolites, MAX_ID_metabolites));  
    
    f_mul = numpy.matlib.repmat(f, 1, MAX_ID_metabolites)
    #s_step =  np.dot(b2m, m2b.T) # s_step is the conversion matrix of each trophic level/layer
    s_step =  np.dot(b2m, f_mul*m2b.T) # s_step is the conversion matrix of each trophic level/layer
    s_step_ii = np.eye(MAX_ID_metabolites, MAX_ID_metabolites)
    #f_mul = numpy.matlib.repmat(f[np.newaxis,:], MAX_ID, 1)#numpy.matlib.repmat(f, 1, MAX_ID)
    #f_mul = numpy.matlib.repmat(f, 1, MAX_ID_metabolites)
    
    """The for loop below iterates over mulitple levels of secretion and consumption across trophic levels, such that the first level of m2m_layer is an identity matrix and each subsequent layer adds the dot product of the conversion matrix with the previous layer of m2m_layer. m2b_total stores the cumulative secretion over every layer, al of which multiplied with the last step of biomass accumulation outside the for loop.
    For the limiting case of a single trophic layer, this means that m2m_layer and m2b_total will just be the identity matrix at the end of the for loop, because s_step_ii only gets updated at the end of every iteration and the for lopp only runs once for numLevels_max = 1. But the logic is that m2m_layer should finally have the net secretion for each level obtained from the dot product of b2m and f*m2b.T, and m2b_total is the resultant of this secretion and the final step of biomasss accumulation, which is given by (1-f)*m2b. The resultant is calculated using a dot product exactly like the first step secretion, where m2b_total till then gives the total secretion (like b2m) and (1-f)*m2b.T gives the biomass accumulation (as opposed to f*m2b.T which gives the secreted fraction of the uptake.)"""
    for ii in range(numLevels_max):
        s_step_ii = np.dot(s_step_ii, s_step)
        m2b_total = m2b_total + s_step_ii
        m2m_layer[:,:,ii] = s_step_ii
        # s_step_ii = np.dot(s_step_ii, s_step)  
    m2m_total = m2b_total
    m2b_total = np.dot((1 - f_mul) * m2b.T, m2b_total) # m2b_total has an extra multiplication of m2b and (1-f).
    return [m2b_total, m2m_total, m2m_layer]

def pred_error(ct_hyp, net, numLevels_max, f, x, ec_real):
    '''
    pred_error is a function used to compute the logarithmic error between experimentally measured
    metagenome and predicted metagenome computed from the model for a certain nutrient intake. It relies on 
    (1) x: the nutrient intake, (2) i_intake: IDs of the nutrient intake, (3) m2b_total: a conversion matrix 
    from the nutrient intake to the total biomass, and (4) ct_hyp: hypothesised relative cell type frequenices. The 
    first three is used to compute the net gain in the intracellular metabolome predicted by the model "ic_pred" and compare it with the 
    experimentally measured net gain in intracellular metabolome "ic_real".
    '''
    x = np.clip(x, 0, np.inf)
    m2b, b2m = Ain_out(ct_hyp, x, net)
    m2b_total, m2m_total, m2m_layer = m2b_multiple_levels(f, m2b, b2m, numLevels_max)
    
    ec_pred = m2m_total.sum(0) # Row sum of the final secretion matrix gives the extracellular metabolome since cell type abundances and relative abundances in the diet have been accounted in previous steps already.
    
    pred_error = (np.log10(ec_pred + 1e-6) - np.log10(ec_real +1e-6)) / np.log10(ec_real +1e-6)
    pred_error = np.sqrt(np.dot(pred_error, pred_error.T)) #np.sqrt(np.sum(pred_error**2))
    return pred_error

def calc_metabolome(x, m2b, m2m_total, numLevels_max):
    '''
    calc_metabolome is a function used to calculate the metabolome from the fitted nutrient intake from the
    model. It relies on (1) x: the nutrient intake, (2) i_intake: IDs of the nutrient intake, (3) m2m_layer: 
    a conversion matrix from the nutrient intake to the metabolite byproducts at a trophic level or layer, 
    and (4) numLevels_max: the number of trophic levels/layers in the model. The metabolome in the model is 
    assumed to be composed of two parts: (1) met_levels: all metabolites in the final trophic level/layer 
    (which is considered to be reaching the end of the gut because of the finite gut length and gut motility.),
    and (2) met_leftover_levels: all unusable metabolites from all previous trophic levels/layers. 
    '''
    i_unused = np.where(np.sum(m2b.T,0) == 0)[0]
    met_levels = np.zeros((MAX_ID_metabolites, numLevels_max))  
    met_leftover_levels = np.zeros((MAX_ID_metabolites, numLevels_max))
    
    # x_full = np.zeros((MAX_ID_metabolites,));
    # x_full[i_intake] = x;
    
    # for ii in range(numLevels_max):
    met_levels = m2m_total.sum(0) #np.dot(m2m_layer[:,:,ii], x)
    met_leftover_levels[i_unused, 0] = x[i_unused]
    # if ii==0:
    # met_leftover_levels[i_unused,ii] = x[i_unused]
    # else:
    #     met_leftover_levels[i_unused,ii] = met_levels[i_unused,ii-1]
            
    return [met_levels, met_leftover_levels]

def run_network_model(f, col_name):
    # f_byproduct = 0.9
    f = f * np.ones((MAX_ID_celltypes,1))
    # f[i_selfish] = 0.0;  # The byproduct/leakage fraction f for celltypes that don't generate byproducts is set as 0.

    numLevels_max = 1

    ######## Pull out experimentally measured intracellular metabolite net gain for one cell type
    # pa = 5; 
    # ic_real = ic_metabolome[col_name].to_numpy(dtype=np.float64)
    # ic_real = ic_real/ic_real.sum() # Relative intracellular metabolite abundances
    ct0 = celltypefreq.to_numpy() # Prior cell type relative frequencies
    ec_real = ec_metabolome[col_name].to_numpy(dtype=float)
    # b_real[metagenome_ID] = metagenome.iloc[:,pa] / np.sum(metagenome.iloc[:,pa])

    ######## Assign diet using 0h extracellular measurement from Immanuel et al.
    diet = pd.read_csv('../input-data/diet.csv')
    diet = diet[np.isin(diet['metabolites_ID'], ic_metabolome_ID)]
    x = diet.loc[:, col_name]
    x = np.clip(x, 0, np.inf)
    x = x/x.sum() # Initial relative abundances of metabolites in the diet


    ######## Compute matrices involving the metabolite consumption and generation:
    m2b, b2m = Ain_out(ct0, x, net)
    m2b_total, m2m_total, m2m_layer = m2b_multiple_levels(f, m2b, b2m, numLevels_max)
    #m2b_total, m2m_total, m2m_layer = m2b_multiple_levels(i_nonzero_celltypes, i_nonzero_metabolites, f, m2b, b2m, numLevels_max)

    ######## The model is converted into an optimization problem where the nutrient intake is constantly changed to
    # minimize the logarithmic error between experimentally measured metagenome and predicted metagenome computed 
    # from the model for a certain nutrient intake.
    fun = lambda ct_hyp: pred_error(ct_hyp, net, numLevels_max, f, x, ec_real)
    bnds = ((0, 1), ) * len(ct0)
    constraint = {'type': 'eq', 'fun': lambda ct_hyp: ct_hyp.sum() - 1.}
    # res = minimize(fun, ct0, method='SLSQP', bounds=bnds, options={'disp': True, 'maxiter': 1000}, tol=1e-3)
    # res = minimize(fun, ct0, method='Nelder-Mead', bounds=bnds, options={'disp': True, 'maxiter': 1000}, tol=1e-3)
    res = minimize(fun, ct0, method='trust-constr', bounds=bnds, options={'disp': False, 'maxiter': 1000}, tol=1e-3, constraints=constraint)
    print(res)
    ######## As long as the optimized nutrient intake is found by using the above optimization solver, the
    # optimized nutrient intake "res.x" is used to generate the predict metagenome and metabolome. They are
    # visually and statistically compared to the experimentally measured metagenome and metabolome.
    ct_full = np.zeros((MAX_ID_celltypes,))
    ct_full = res.x
    # ic_pred = np.matmul(m2b_total, ct_full)
    # ec_real = ec_metabolome['NSP']

    # figname = str(f)+'-'+col_name
    # #### Metagenome comparison
    # fig, ax = plt.subplots()
    # ax.loglog(ic_pred, ic_real, 'ko')
    # # ax.plot([1e-5, 1], [1e-5, 1],'k-')
    # ax.set_aspect('equal')
    # ax.set_xlabel('predicted')
    # ax.set_ylabel('experimentally observed')
    # ax.set_title('Intracellular metabolome comparison')
    # fig.savefig('../figures/relative-abund-one-celltype-static-net/ic-pred-correlation-'+figname+'.png', dpi=300.)

    met_levels, met_leftover_levels = calc_metabolome(x, m2b, m2m_total, numLevels_max)
    # metabolome_measured = np.zeros((MAX_ID_metabolites,))
    metabolome_measured = ec_metabolome[col_name].to_numpy(dtype=float)
    # metabolome_measured = metabolome_measured/metabolome_measured.sum()
    #metabolome_measured[metabolome_ID.values] = -0.58888854, -0.55040201metabolome.iloc[:,pa]
    # metabolome_measured[metabolome_ID] = metabolome.iloc[:,pa]
    metabolome_pred = met_levels + met_leftover_levels.ravel() #np.dot(m2m_total, x_full)
    #metabolome_pred = np.dot(m2m_total, x_full)
    i_common = np.where(metabolome_measured * metabolome_pred > 1e-5)[0]
    metabolome_pred_common = metabolome_pred[i_common] / np.sum(metabolome_pred[i_common])
    metabolome_measured_common = metabolome_measured[i_common] / np.sum(metabolome_measured[i_common])

    #### Metabolome comparison
    fig, ax = plt.subplots()
    ax.loglog(metabolome_pred_common, metabolome_measured_common, 'ko')
    ax.plot([1e-5, 1], [1e-5, 1],'k-')
    ax.set_aspect('equal')
    ax.set_xlabel('predicted')
    ax.set_ylabel('experimentally observed')
    ax.set_title('Extracellular metabolome comparison')
    # fig.savefig('../figures/relative-abund-one-celltype-static-net/ec-pred-correlation-'+figname+'.png', dpi=300.)
    

    # ic_corr = pearsonr(ic_pred[ic_real>0], ic_real[ic_real>0])[0]
    ec_corr = pearsonr(metabolome_pred_common, metabolome_measured_common)[0]
    print('-------------------------------------------------------------------------------------------------------------')
    # print('(Correlation coefficient, P-value) of the correlation between predicted and experimentally measured intracellular metabolome:')
    # print(pearsonr(ic_pred[ic_real>0], ic_real[ic_real>0]))
    print('(Correlation coefficient, P-value) of the correlation between predicted and experimentally measured extracellular metabolome:')
    print(pearsonr(metabolome_pred_common, metabolome_measured_common))

    return ec_corr, ct_full


# %%
###### Actually run the model for whatever replicates we have
f_arr = np.linspace(0, 1, 10)
celline_names = ['U87MG', 'NSP']

# ic_corr = np.zeros((len(f_arr), len(celline_names)))
ec_corr = np.zeros((len(f_arr), len(celline_names)))
ct_full = ec_corr.copy()

for i in range(len(f_arr)):
    for j in range(len(celline_names)):
        ec_corr[i, j], ct_full[i, j] = run_network_model(f_arr[i], celline_names[j])


# %%
plt.imshow(ec_corr.T, cmap='viridis')
plt.colorbar()
plt.xticks(ticks=np.arange(len(f_arr)), labels=f_arr.round(decimals=2), rotation=45)
plt.yticks(ticks=[0, 1], labels=celline_names)
plt.title('Pearsons r: Extracellular conc')
plt.xlabel('Byproduct fraction, f')
plt.ylabel('Cell line')
plt.savefig('../figures/relative-abund-one-celltype-static-net/extracellular-relative-conc-pearsonr.png', dpi=300.)

# %%
plt.imshow(ct_full.T, cmap='viridis')
plt.colorbar()
plt.xticks(ticks=np.arange(len(f_arr)), labels=f_arr.round(decimals=2), rotation=45)
plt.yticks(ticks=[0, 1], labels=celline_names)
plt.title('Cell type abundance')
plt.xlabel('Byproduct fraction, f')
plt.ylabel('Cell line')
plt.savefig('../figures/relative-abund-one-celltype-static-net/celltype-abundance.png', dpi=300.)

####################################################################
# %%
