# Differential functional reorganization of ventral and dorsal visual pathways following childhood hemispherectomy

Accompanying code and data repository for: 

Ayzenberg, V., Granovetter, M. C., Robert, S., Patterson, C., & Behrmann, M. (2023). Differential functional reorganization of ventral and dorsal visual pathways following childhood hemispherectomy. bioRxiv.


### All main-text figures and tables can be recreated in the following notebooks:
create_plots: Creates all summary plots including summed selectvitiy, decoding, and confound metrics
create_neural_figs: Creates all anatomical and functional brain plots included in the manuscript
create_neural_heatmap: Creates 2D heatmap plots of neural data for dorsal and ventral pathways
create_tables: Creates tables of all the data and analyses in the manuscript


### All data used to conduct the analyses and create the figures can be found in the results folder

Within results/

selectivity - has the summary file for all selectivity metrics: mean activation, activate volume, summed selectivity, and normalized summed selectivity (sum_selec_norm). The primary analyses in the manuscript are based on the normalized summed selectivity values

decoding - summary file for decoding analyses, oftened simply labled as 'acc' for decoding accuracy.

neural map - data for the location of patients' and controls' peak responses

confound - contains summaries for all confound metrics

Each folder also contains resamples of the control data for non-parametric statistics

#### analyses folder contrains all scripts for computing summary values from fMRI data

#### fMRI folder contains preprocessing and signal extraction scripts

#### figures contains all figures used in the manuscript

#### behavior contains raw behavioral data, as well as notebooks for analysis and plotting







