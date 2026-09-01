# cancer-recon-network
A network-based resource consumption model of metabolic strategies in cancer

## In ```codes/```
- ```cancer-power-law.py``` has all the codes for data import, filtering, analysis and plotting for Figure 1.
- ```cancer-data-processing.py``` has all the code used for initial data filtering and generating the input pickle for all further simulations. This needs to be run only once for a given set of filtering criteria. For a given number of cell phenotypes (set in line 150 here), the script outputs a pickle into the same directory that is then accessed by other downstream scripts below.
- ```cancer-no-learning-networks.py``` is the script for generating balanced networks from a given random network. This is where all the data are generated for Figures 2 and 3.
- ```cancer-all-network-models.py``` is the script for adding all individual overlaps to a balanced initial network without overlap, as well as the algorithmic learning process for network improvement. This produces the data used in Figure 4.
- ```figure-plan.py``` contains plotting codes for all figures from Figure 2 onwards.