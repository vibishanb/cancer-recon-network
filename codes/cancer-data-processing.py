# %% [markdown]
# # Data processing for cancer uptake-secretion model
# This file is used to pre-process all data (especially Chia network and Thai Children data) into the format which is convenient for simulations.

# %%
########### Self-customized setting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
########### Load names of all nodes in the prior network
names = pd.read_csv('../input-data/jain-data/names_ID.txt',sep=':')
names.set_index('IDs', inplace=True)
print(names.head())

########### Load names of all nodes in the prior network
i_intake = pd.read_csv('../input-data/jain-data/nutrient_intake_ID.txt',sep=':')
i_intake = i_intake['IDs'].values
print(i_intake)

########### Load mean cell abundance priors-randomly sampled from a uniform distribution ranged [0, 1)
k = 1 # Assumed number of cell types for current iteration of the model
celltype_all = pd.read_csv('../input-data/prior-celltype-abundance.txt', sep=',')
celltype_all = celltype_all.iloc[:k, ] # Selecting number of cell types
celltype_all.head()
### Randomly sampled relative abundances of the cell types
rand = np.random.uniform(0, 1, k)
celltype_all['Mean'] = rand/rand.sum()
celltype_ID = celltype_all['celltype_id']
#print((celltype_ID!=0).sum())
celltype = celltype_all[celltype_ID!=0].loc[:,'Mean']
celltype_ID = celltype_ID[celltype_ID!=0]
print(celltype.head())

# %%
########### Load extracellular metabolome for all cell lines
ec_metabolome_all = pd.read_excel('../input-data/jain-data/metabolome-jain.xlsx', sheet_name='ec_metabolites')
core_data_all = pd.read_excel('../input-data/jain-data/metabolome-jain.xlsx', sheet_name='CORE_profile')

i_valid_mets = np.where(ec_metabolome_all['Calibrated'] > 0 )[0]
ec_metabolome = ec_metabolome_all.iloc[i_valid_mets, :]

i_valid_cell_lines = np.where(np.isin(core_data_all.columns, ec_metabolome.columns))[0]
core_data = core_data_all.iloc[i_valid_mets, i_valid_cell_lines]
ec_metabolome_ID = ec_metabolome['metabolites_ID']
print(ec_metabolome.head())
print('-----------------------------------------------')
print(core_data.head())


# %%
################# Mass balance checks on the ec-metabolome
met_test = ec_metabolome.copy()
mw = met_test.iloc[:, 3]
met_test = met_test.iloc[:, 4:]
met_test.columns = met_test.columns.str.split('.').str[0]
met_test = met_test.T.reset_index(names=['Cell line'])
met_test_mean = met_test.groupby(['Cell line']).mean().T

met_baseline = met_test_mean.iloc[:, np.isin(met_test_mean.columns, ['Baseline1', 'Baseline2', 'Baseline3', 'Baseline4', 'Baseline5'])]
met_test_mean = met_test_mean.iloc[:, ~np.isin(met_test_mean.columns, ['Baseline1', 'Baseline2', 'Baseline3', 'Baseline4', 'Baseline5'])]

baseline_mass = met_baseline.mul(mw, axis=0).sum(axis=0).to_numpy()/10**6
test_mass = met_test_mean.mul(mw, axis=0).sum(axis=0).to_numpy()/10**6
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
f, ax = plt.subplots()
sns.scatterplot(test_mass, ax=ax)
ax.hlines(y = baseline_mass, xmin=0, xmax=60, linestyles='dashed', colors='r', label='Baseline')
ax.hlines(y = baseline_mass.mean(), xmin=0, xmax=60, linestyles='solid', colors='k', lw=2, label='Mean baseline')
plt.legend(loc='lower right')


# %%
########### Generate (containing information of metabolite consumption and production)
net = pd.read_csv('../input-data/jain-data/prior-recon-network.csv').dropna()
net = net.iloc[i_valid_mets, :]
# net.loc[:, 'metabolites_ID'] = net.loc[:, 'metabolites_ID'].to_numpy().astype(str)
# mean_net = net.groupby('celltypes_ID').mean()
# valid_net = net[net.iloc[:, 3] != 0] # No transport reactions found
valid_net = net[net['celltypes_ID'] <= k] # Network with selected number of cell types
valid_net = valid_net[np.isin(valid_net.loc[:, 'metabolites_ID'], ec_metabolome_ID)]
print(valid_net.head())

# %%
########### Process CORE profiles-average over duplicates and arrange with baseline vs spent medium
core_data.columns = core_data.columns.str.split('.').str[0]

mets_list = core_data.loc[:, 'metabolite_name'] #Saving the order of metabolites for future reference
core_data = core_data.iloc[:, 2:] # Retaining only CORE profile data and cell line names

core_data = core_data.T.reset_index(names=['Cell line'])
core_mean = core_data.groupby(['Cell line']).mean().reset_index() # Average CORE profile for each cell type, mean over duplicates. Each row is a cell type, and columns are mets. This way, exporting to ndarray gives an array of 60 arrays, each with the CORE profile for one cell line.
print(core_data.head())
print("-----------------------------------------------------")
print(core_mean.head())

i_intake = i_intake[np.isin(i_intake, ec_metabolome_ID)]
names = names[np.isin(names.index, np.append(['1', '2', '3', '4', '5'], ec_metabolome_ID.to_numpy()))]

# %%
############### Initial uptake-secretion network based on CORE profiles
n_lines = len(core_mean.index)
all_networks = []
for i in range(n_lines):
    curr_net = valid_net.copy()
    curr_net.iloc[:, -1] = np.where(core_mean.iloc[i, 1:] == 0, 0, 5)
    all_networks.append(curr_net)


# %%
########### pickle all processed data which are useful for simulations
import pickle

pickle_out = open("cancer_network.pickle","wb")
#pickle.dump([net, i_selfish, i_intake, names], pickle_out)
pickle.dump([all_networks, i_intake, names], pickle_out, protocol=2)
pickle_out.close()

pickle_out = open("data.pickle","wb")
pickle.dump([celltype_ID, celltype, ec_metabolome_ID, met_test_mean, met_baseline, core_mean], pickle_out, protocol=2)
pickle_out.close()



# %%
