import cellCnn
import importlib
import numpy as np
### (from: https://github.com/eiriniar/CellCnn/blob/0413a9f49fe0831c8fe3280957fb341f9e028d2d/cellCnn/examples/NK_cell_ungated.ipynb ) AND https://github.com/eiriniar/CellCnn/blob/0413a9f49fe0831c8fe3280957fb341f9e028d2d/cellCnn/examples/PBMC.ipynb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt

from cellCnn.ms.utils.helpers import get_min_mean_from_clusters, get_chunks
from cellCnn.utils import mkdir_p
from cellCnn.plotting import plot_results
from cellCnn.ms.utils.helpers import get_chunks
from cellCnn.ms.utils.helpers import print_regression_model_stats
from cellCnn.plotting import plot_results
from cellCnn.utils import mkdir_p
from cellCnn.utils import save_results, get_selected_cells, get_data
from cellCnn.ms.utils.helpers import get_fitted_model
from cellCnn.ms.utils.helpers import split_test_valid_train
from cellCnn.ms.utils.helpers import calc_frequencies, get_chunks_from_df

import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
#%%
#%reset
#%%
##### state vars
cytokines = ['CCR2', 'CCR4', 'CCR6', 'CCR7', 'CXCR4', 'CXCR5', 'CD103', 'CD14', 'CD20', 'CD25', 'CD27', 'CD28', 'CD3',
             'CD4', 'CD45RA', 'CD45RO', 'CD56', 'CD57', 'CD69', 'CD8', 'TCRgd', 'PD.1', 'GM.CSF', 'IFN.g', 'IL.10',
             'IL.13', 'IL.17A', 'IL.2', 'IL.21', 'IL.22', 'IL.3', 'IL.4', 'IL.6', 'IL.9', 'TNF.a']
infile = 'cohort_denoised_clustered_diagnosis_patients.csv'
indir = 'data/input.nosync'
outdir = 'class'
rand_seed = 123
train_perc = 0.7
test_perc = 0.3
batch_size_pheno = 840  # deprecated  # so a size of 8425 is about equally sized in batches
batch_size_cd4 = 550  # deprecated #so a size of 550 gets me 16 batches for cd4
## information from ms_data project
cluster_to_celltype_dict = {0: 'b', 1: 'cd4', 3: 'nkt', 8: 'cd8', 10: 'nk', 11: 'my', 16: 'dg'}
outfolder = 'mtl_1'
mkdir_p(outfolder)

#%%

np.random.seed(rand_seed)
mkdir_p(outdir)
df = pd.read_csv(os.path.join(indir, infile), index_col=0)
df = df.drop_duplicates()  ### reduces overfitting at cost of fewer data
df.shape
##### no duplicates in

#%% ### split by diagnosis state

# pitch: key = gate_source, value = dict{diagnosis: df OR freq?}
rrms_df = df[df['diagnosis'] == 'RRMS']
rrms_patients2df = {id: patient_df.drop(columns=['diagnosis', 'gate_source']) for id, patient_df in
                    rrms_df.groupby('gate_source')}
nindc_df = df[df['diagnosis'] == 'NINDC']
nindc_patients2df = {id: patient_df.drop(columns=['diagnosis', 'gate_source']) for id, patient_df in
                     nindc_df.groupby('gate_source')}

#%% ### celltype frequencies

#### here we could see freq differences across the 2 groups
print('Frequencies: ')
rrms_patients_freq = {id: calc_frequencies(patient_df, cluster_to_celltype_dict, return_list=True) for id, patient_df in
                      rrms_patients2df.items()}
nindc_patients_freq = {id: calc_frequencies(patient_df, cluster_to_celltype_dict, return_list=True) for id, patient_df
                       in nindc_patients2df.items()}
print('DONE')
# we got 31 patients each

#%% ### Also ich brauch nicht mehr nach cell types in den batches suchen, es ist recht egal.

# todo ich muss eigentlich die frequency pro batch berechnen...
importlib.reload(cellCnn.ms.utils.helpers)
from cellCnn.ms.utils.helpers import *

batch_size_dict = dict()
### desease states 1 = RRMS and 0 = NINDC
selection_pool_rrms_cd8 = get_chunks_from_df(rrms_patients2df,
                                             freq_df=rrms_patients_freq,
                                             desease_state=1)
selection_pool_nindc_cd8 = get_chunks_from_df(nindc_patients2df,
                                              freq_df=nindc_patients_freq,
                                              desease_state=0)

# make sure list are equally long:
if len(selection_pool_rrms_cd8) > len(selection_pool_nindc_cd8):
    selection_pool_rrms_cd8 = selection_pool_rrms_cd8[:len(selection_pool_nindc_cd8)]
elif len(selection_pool_rrms_cd8) < len(selection_pool_nindc_cd8):
    selection_pool_nindc_cd8 = selection_pool_nindc_cd8[:len(selection_pool_rrms_cd8)]

all_chunks = selection_pool_rrms_cd8 + selection_pool_nindc_cd8
np.random.shuffle(all_chunks)

X = [selection[0].to_numpy() for selection in all_chunks]
freqs = [selection[1] for selection in all_chunks]
Y = [selection[2] for selection in all_chunks]

print('DONE: batches created')

#%%

# Default values were used for most CellCNN hyperparameters,
# except for the following: ncell = 2,000; scale = false;
# maxpool percentages = (0.5, 1.0, 2.5, 5.0, 10.0). T
print('RUN: STL class phenotype:')
print('Galli + default parameters')
from cellCnn import loss
from cellCnn.loss import *
import cellCnn
import cellCnn.utils
import cellCnn.loss

importlib.reload(cellCnn.ms.utils.helpers)
importlib.reload(cellCnn.utils)
importlib.reload(cellCnn.model)
importlib.reload(cellCnn.loss)
importlib.reload(cellCnn)
from cellCnn.loss import *
from cellCnn.ms.utils.helpers import *
import cellCnn
import cellCnn.utils
from cellCnn.loss import RevisedUncertaintyLoss

# for regression task stratified is wrong since there are no classes
freq_idx = 3
cd8 = [series[freq_idx] for series in freqs]
X_test, X_train, X_valid, cd8_test, cd8_train, cd8_valid, y_test, y_train, y_valid = split_test_train_valid(
    X, cd8, Y,
    train_perc=train_perc,
    test_perc=test_perc,
    valid_perc=0.5)

# pay attention to put in phenotype first and then freq...
model = get_fitted_model(X_train, X_valid, y_train, y_valid,
                         nsubset=1000, ncell=2000,
                         quant_normed=True, scale=False,
                         nfilters=[3, 7, 20, 35], max_epochs=20,
                         learning_rate=None, nrun=15,
                         per_sample=True, regression=False,
                         maxpool_percentages = [0.5, 1.0, 2.5, 5.0, 10.0],
                         outdir=outdir, verbose=True, mtl_tasks=1)
print('DONE building models')

#%%

from plotting import plot_filters, discriminative_filters

results = model.results

# exported_filter_weights: consensus filtzer from results['selected_filters'], filters_all from results['clustering_result']
save_results(results, outdir, cytokines)

#%%

plotdir = os.path.join(outdir, 'plots')

# consensus_filter_weights, best_net, clustered_filter_weights
plot_filters(results, cytokines, os.path.join(plotdir, 'filter_plots'))

# filter response difference.pdf -> only for consensus filters
_v = discriminative_filters(results, os.path.join(plotdir, 'filter_plots_discriminative'),
                            filter_diff_thres=0.2, show_filters=True)
print('done')

#%%

# just took default values
# colors in plots are coming from the sample input (here the X_train)
filter_info = plot_results(results, X_train, y_train,
                           cytokines, os.path.join(plotdir, 'training_plots'),
                           filter_diff_thres=0.2,
                           filter_response_thres=0,
                           stat_test='mannwhitneyu',
                           group_a='RRMS', group_b='NINDC',
                           group_names=['Test_a', 'test_b'],
                           tsne_ncell=1000,
                           regression=False,
                           show_filters=True)
print('done')

#%%

_v = plot_results(results, X_valid, y_valid,
                  cytokines, os.path.join(plotdir, 'validation_plots'),
                  filter_diff_thres=0.2,
                  filter_response_thres=0,
                  stat_test='mannwhitneyu',
                  group_a='RRMS', group_b='NINDC',
                  group_names=['Test_a', 'test_b'],
                  tsne_ncell=1000,
                  regression=False,
                  show_filters=True)
csv_dir = os.path.join(outdir, 'selected_cells')
mkdir_p(csv_dir)
nfilter = len(filter_info)

# args.fcs = des gibt mir die fcs folder ich hol mir in sample names also die verschiedenen fcs filenames raus...
# ich arbeite nur mit einem fcs file  (ich nutze ja nur den cohort)
# drum verteil ich den name jetzt einfach statisch
# fcs_info = np.array(pd.read_csv(args.fcs, sep=','))
# sample_names = [name.split('.fcs')[0] for name in list(fcs_info[:, 0])]

#%%

sample_names = ['cohort']
df2 = pd.read_csv(os.path.join(indir, infile), index_col=0)
df2 = df2.drop_duplicates()
test = df2.iloc[:, :len(cytokines)]
samples = [test]
# for each sample
for x, x_name in zip(samples, sample_names):
    flags = np.zeros((x.shape[0], 2 * nfilter))
    columns = []
    # for each filter
    for i, (filter_idx, thres) in enumerate(filter_info):
        flags[:, 2 * i:2 * (i + 1)] = get_selected_cells(
            results['selected_filters'][filter_idx], x, results['scaler'], thres, True)
        columns += ['filter_%d_continuous' % filter_idx, 'filter_%d_binary' % filter_idx]
    df = pd.DataFrame(flags, columns=columns)
    df.to_csv(os.path.join(outdir, x_name + '_selected_cells.csv'), index=False)

print('done')


#%%

import cellCnn
from cellCnn.model import CellCnn
import cellCnn.model

importlib.reload(cellCnn)
importlib.reload(cellCnn.model)
import cellCnn
from cellCnn.model import CellCnn
import cellCnn.model

######################################################################
# what predict does is: Iterating over the best 3 nets and return an array per net with the predictions for all samples!
stats_file = open(f"{outdir}/stats_class_{model.ncell}_{model.nsubset}.txt", "w+")
print('Write stats to: \n')
print(stats_file.name)
stats_file.write(f"Model {outdir} with ncells: {model.ncell} and subsets: {model.nsubset}.txt")

test_pred = model.predict(X_test, mtl_inputs=[y_test])
train_pred = model.predict(X_train, mtl_inputs=[y_train])
valid_pred = model.predict(X_valid, mtl_inputs=[y_valid])
#print_regression_model_stats(test_pred, b_test)
test_pred_abs = [1 if pred[0] > pred[1] else 0 for pred in test_pred]
train_pred_abs = [1 if pred[0] > pred[1] else 0 for pred in train_pred]
valid_pred_abs = [1 if pred[0] > pred[1] else 0 for pred in valid_pred]

acc_test = accuracy_score(list(y_test), test_pred_abs)
acc_train = accuracy_score(list(y_train), train_pred_abs)
acc_valid = accuracy_score(list(y_valid), valid_pred_abs)
stats_file.write('Desease: \n')
stats_file.write(f'Acc y test {acc_test}\n')
stats_file.write(f'Acc y train {acc_train}\n')
stats_file.write(f'Acc y valid {acc_valid}\n')
stats_file.close()
print('DONE')