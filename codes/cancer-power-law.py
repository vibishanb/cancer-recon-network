"""
<!---------------------------
Name: cancer-recon-network
File: cancer-power-law
-----------------------------
Author: bvibishan
Data:   07/10/2025, 17:16:14
---------------------------->
"""
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import forestplot as fp

from scipy.stats import linregress, t
import os
sns.set_context('notebook')
# %%
###### Importing actual metabolome data
ec_metabolome_all = pd.read_excel('../input-data/jain-data/metabolome-jain.xlsx', sheet_name='ec_metabolites')
i_valid_mets = np.where(ec_metabolome_all['Calibrated'] > 0 )[0]
valid_met_IDs = ec_metabolome_all['metabolites_ID'].iloc[i_valid_mets].to_numpy()
ec_metabolome = ec_metabolome_all.iloc[i_valid_mets, :]

met_test = ec_metabolome.copy()
mw = met_test.iloc[:, 3]
met_test = met_test.iloc[:, 4:]
met_test.columns = met_test.columns.str.split('.').str[0]
met_test = met_test.T.reset_index(names=['Cell line'])
met_test_mean = met_test.groupby(['Cell line']).mean().T

leukemia_cell_lines = ['CCRF-CEM', 'HL-60-TB', 'K562', 'MOLT-4', 'RPMI-8226', 'SR']
filtered_cell_lines = np.concatenate([leukemia_cell_lines, ['Baseline1', 'Baseline2', 'Baseline3', 'Baseline4', 'Baseline5']])
ec_metabolome = met_test_mean.iloc[:, ~np.isin(met_test_mean.columns, filtered_cell_lines)] # Final df of measured metabolome post growth

met_baseline = met_test_mean.iloc[:, np.isin(met_test_mean.columns, ['Baseline1', 'Baseline2', 'Baseline3', 'Baseline4', 'Baseline5'])]
diet = met_baseline.mean(axis=1) # Average diet

# %%
slopes = []
slopes_err = []
slopes_ci = []
rmse_ref = []
rmse_fitted = []
tinv = lambda p, df: abs(t.ppf(p/2, df)) # Lambda function to calculate CI

fig_path = '../figures/power-law-plots/diet-scatter/'
try:
    os.makedirs(fig_path)
except:
    pass
figsave_flag = True

for cl in ec_metabolome.columns.values:
    ec_measured = ec_metabolome.loc[:, cl]
    i_common = np.where(ec_measured * diet > 0)[0] # Take only non-zero mets

    x = np.log10(diet.values[i_common])
    y = np.log10(ec_measured.values[i_common])
    res = linregress(x, y)
    slopes.append(res.slope)
    slopes_err.append(res.stderr)
    ts = np.abs(tinv(0.05, len(x)-1)) # CI magnitude
    slopes_ci.append(ts * res.stderr)

    y_pred = (res.slope * x) + res.intercept # Predicted metabolome based on the diet
    ## RMSE for the fitted least-squares line
    diff_fitted = y_pred - y
    rmse_fitted.append(np.sqrt(np.mean(diff_fitted**2)))

    ## RMSE for the reference line with slope 1 and intercept 0 => predicted value is simply the diet
    diff_ref = x - y
    rmse_ref.append(np.sqrt(np.mean(diff_ref**2)))

    plt.scatter(x, y, c='k', s=50)
    plt.plot(x, y_pred, 'r-', linewidth=2.5)
    plt.plot(x, x, 'g', linestyle='dashed', linewidth=2.5)
    plt.xlabel(r'$log_{10}\ Diet$')
    plt.ylabel(r'$log_{10}\ Measured\ metabolome$')
    plt.title(cl, fontsize=20)
    plt.text(x.min(), y.max()-0.2,
             f'Measured slope = {res.slope:.2f}', fontsize=15, c='r')
    plt.text(x.min(), y.max()-1,
             f'Reference slope = 1', fontsize=15, c='g')
    
    if figsave_flag:
        plt.savefig(fig_path+'diet-scatter-'+cl+'.png', dpi=300)
        plt.close()
    else:
        plt.show()

slopes = np.array(slopes)
slopes_err = np.array(slopes_err)
slopes_ci = np.array(slopes_ci)
rmse_fitted = np.array(rmse_fitted)
rmse_ref = np.array(rmse_ref)

# %%
slope_df = pd.DataFrame({'Cell line': ec_metabolome.columns.values, 'Slope': slopes,
                         'SE': slopes_err, 'CI': slopes_ci,
                         'Smin': slopes - slopes_ci, 'Smax': slopes + slopes_ci,
                         'RMSE_Fitted': rmse_fitted, 'RMSE_Ref': rmse_ref,
                         'PowerLaw': np.where(slopes + slopes_ci < 1, 'Sublinear', 'Linear')})
## Distribution of slopes
fig_path = '../figures/power-law-plots/'
sns.histplot(data=slope_df, x='Slope', hue='PowerLaw',
             bins=12, stat='density',  element='step', 
             alpha=0.65, fill=True, multiple='stack', palette='crest')
plt.title('Diet vs metabolome slope')
if figsave_flag:
    plt.savefig(fig_path+'distribution-of-slopes.png', dpi=300)

# %%
## CIs of slopes and RMSE-forest plots
slope_df.loc[:, 'Del_RMSE'] = slope_df.loc[:, 'RMSE_Fitted'] - slope_df.loc[:, 'RMSE_Ref']

# f, ax = plt.subplots(1, 2)
ax = fp.forestplot(slope_df,  # the dataframe with results data
              estimate="Slope",  # col containing estimated effect size 
              ll="Smin", hl="Smax",  # columns containing conf. int. lower and higher limits
              varlabel="Cell line",  # column containing variable label
              color_alt_rows=True,
              xlabel=r'Slope $\pm$ 95% CI',
              ci_report=True, flush=False,
              sort=True, sortby='Smax',
              figsize=(4, 16))
ax.axvline(x=1, ymax=0.96, c='r', linestyle='dashed', linewidth=2)
plt.savefig(fig_path+'slopes-forest-plot.png', bbox_inches='tight', dpi=300)

ax = fp.forestplot(slope_df,  # the dataframe with results data
              estimate="Del_RMSE",  # col containing estimated effect size 
              varlabel="Cell line",
              ylabel="",
              flush=False,
              color_alt_rows=True,
              xlabel=r'$\Delta$ RMSE = Fitted $-$ Reference',
              ci_report=False,
              sort=True, sortby='Smax',
              xticks=[-0.1, -0.075, -0.05, -0.025, 0, 0.025],
              figsize=(5, 15))
plt.savefig(fig_path+'del-rmse-forest-plot.png', bbox_inches='tight', dpi=300)

# %%
i_sublinear = np.where(slope_df.loc[:, 'Smax'] < 1)[0]
print({slope_df.iloc[i_sublinear, 1].mean()})
print({slope_df.iloc[i_sublinear, 1].std()})

sublinear_cell_lines = ec_metabolome.columns.values[i_sublinear]

########### pickle all processed data which are useful for simulations
import pickle

pickle_out = open("cancer_power_law_stats.pickle","wb")
#pickle.dump([net, i_selfish, i_intake, names], pickle_out)
pickle.dump([slope_df, sublinear_cell_lines], pickle_out, protocol=2)
pickle_out.close()
# %%
