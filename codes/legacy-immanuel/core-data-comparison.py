"""
<!---------------------------
Name: cancer-recon-network
File: core-data-comparisons
-----------------------------
Author: bvibishan
Data:   6/25/2025, 5:33:00 PM
---------------------------->
"""
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette('viridis')
sns.set_context('talk')

# %%
#### Jain et al (2012)
ec_metabolome_all = pd.read_excel('../input-data/jain-data/metabolome-jain.xlsx', sheet_name='ec_metabolites')
core_data_all = pd.read_excel('../input-data/jain-data/metabolome-jain.xlsx', sheet_name='CORE_profile')

i_valid_mets = np.where(ec_metabolome_all['Calibrated'] > 0 )[0]
valid_met_IDs = ec_metabolome_all['metabolites_ID'].iloc[i_valid_mets].to_numpy()
ec_metabolome = ec_metabolome_all.iloc[i_valid_mets, :]

i_valid_cell_lines = np.where(np.isin(core_data_all.columns, ec_metabolome.columns))[0]
core_data = core_data_all.iloc[i_valid_mets, i_valid_cell_lines]

core_data.columns = core_data.columns.str.split('.').str[0]

mets_list = core_data.loc[:, 'metabolite_name'] #Saving the order of metabolites for future reference
core_data = core_data.iloc[:, 2:] # Retaining only CORE profile data and cell line names

core_data = core_data.T.reset_index(names=['Cell line'])
core_mean = core_data.groupby(['Cell line']).mean().reset_index() # Average CORE profile for each cell type, mean over duplicates. Each row is a cell type, and columns are mets. This way, exporting to ndarray gives an array of 60 arrays, each with the CORE profile for one cell line.

i_excess_mass = np.where(test_mass > baseline_mass.min())[0]
excess_mass_cell_lines = met_test_mean.columns[i_excess_mass]

met_test_mean = met_test_mean.copy().drop(columns=excess_mass_cell_lines)

i_excess_mass_rows = np.where(np.isin(core_mean.iloc[:, 0], excess_mass_cell_lines))[0]
core_mean = core_mean.copy().drop(index=i_excess_mass_rows)

print("These cell lines have been removed for violating mass conservation: %s" % excess_mass_cell_lines.to_numpy())