import shutil
import pickle

import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from cellCnn.model import *



def plot_barplot_for_data(df, savename='default.png', column_to_plot_for='freq_r2', title='title', ylabel='ylabel',
                          sort_ascending=True):
    ct_file_stats = df.sort_values(by=[column_to_plot_for], ascending=sort_ascending)
    ct_file_stats['hue'] = ['revised uncertainty' if len(str.split('_')) == 3 else str.split('_')[-1] for str in
                            list(ct_file_stats.index)]

    plt.style.use(['default'])
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 11}

    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax = sns.barplot(x=ct_file_stats.index, y=column_to_plot_for, hue='hue', data=ct_file_stats)
    labels_nicer = [str.split('_')[1:] for str in list(ct_file_stats.index)]
    new_labels = []
    for lbl in labels_nicer:
        lbl = [lst.replace('reg', 'regression') for lst in lbl]
        lbl = ' '.join(lbl)
        new_labels.append(lbl)
    ax.set_xticklabels(new_labels,
                       rotation=25,
                       ha='right',
                       fontsize=12)
    plt.legend(loc='upper left' if sort_ascending else 'upper right')

    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=16)
    plt.savefig(savename, dpi=150)
    plt.close()


def predict_from_loaded_result(results, X_test, mtl_inputs, regression, loss_mode, nmark=35,):
    accuracies = results['accuracies']
    sorted_idx = np.argsort(accuracies)[::-1][:3]
    n_classes = results['n_classes']
    config = results['config']
    y_pred = dict()
    for irun, idx in enumerate(sorted_idx):
        nfilter = config['nfilter'][idx]  # this is always 4 ...
        maxpool_percentage = config['maxpool_percentage'][idx]
        ncell_per_sample = np.min([x.shape[0] for x in X_test])
        ncell_pooled = max(1, int(maxpool_percentage / 100. * ncell_per_sample))
        model = build_model(ncell_per_sample, nmark,
                            nfilter=nfilter, coeff_l1=0, coeff_l2=0,
                            k=ncell_pooled, dropout=False, dropout_p=0,
                            regression=regression, n_classes=n_classes,
                            lr=0.01, mtl_tasks=len(mtl_inputs), loss_mode=loss_mode)
        weights = results['best_3_nets'][irun]
        if len(weights) != len(model.weights):
            print('DEBUG')
        model.set_weights(weights)
        # select a random subset of `ncell_per_sample` and make predictions
        new_samples = [shuffle(x)[:ncell_per_sample].reshape(1, ncell_per_sample, nmark)
                       for x in X_test]
        data_converted = np.vstack(new_samples)
        data_converted = np.asarray([np.asarray(x).astype(np.float64) for x in data_converted])
        mtl_inputs_feed = []
        for i, task in enumerate(mtl_inputs):
            if i == 0 and not regression:
                # classification
                nclasses = len(np.unique(task))
                mtl_inputs_feed.append(tf.keras.utils.to_categorical(task, nclasses))
            else:
                # if i != 0 its definately a regression task
                mtl_inputs_feed.append(np.asarray(task).astype(np.float64))
        prediction = model.predict([data_converted, *mtl_inputs_feed])
        y_pred[irun] = prediction
    if len(mtl_inputs) == 1:
        # default mode
        results_as_np = np.array(list(y_pred.values()))
        result = np.mean(results_as_np, axis=0)
    else:
        # MTL-mode: take the mean of the prediction of every task
        result = []
        for task_nr in range(len(mtl_inputs)):
            to_mean = [pred[task_nr] for pred in y_pred.values()]
            mean = np.mean(np.array(to_mean), axis=0)
            result.append(mean)
    return result

def plot_heatmap_overlaps(df, overlap_dir_outer, label='Regression MTL selection vs STL frequency tasks', savename='heatmap_overlaps_default.png'):
    plt.figure(figsize=(15, 15))
    g = sns.heatmap(df, xticklabels=True, yticklabels=True,
                    cmap="YlGnBu", fmt='g', linewidths=0.2,
                    annot=True, annot_kws={"fontsize": 10}, )
    g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=10)
    g.set_yticklabels(g.get_yticklabels(), rotation=45, fontsize=10)
    plt.title(label)
    plt.savefig(os.path.join(overlap_dir_outer, savename))
    plt.close()

def add_celltype_evaluation(filter_overview, cluster_to_celltype_dict):
    intersting_filters = dict()
    very_intersting_filters = dict()
    for idx, filter in filter_overview.iterrows():
        abu_deltas = dict(eval(filter['xabu_deltas']))
        for ct_int, ct_delta in abu_deltas.items():
            ct_string = cluster_to_celltype_dict[ct_int]
            ct_abu = round(filter[ct_string], 3)
            append_string = f'{idx};{ct_string};{ct_abu}'
            if ct_int == 1:  #cd4  highest abu
                if ct_delta > 0.45:
                    very_intersting_filters[idx] = append_string
                if ct_delta > 0.35:
                    intersting_filters[idx] = append_string
            elif ct_int in [0, 8]:  #middle-high abu
                if ct_delta > 0.80:
                    very_intersting_filters[idx] = append_string
                if ct_delta > 0.60:
                    intersting_filters[idx] = append_string
            else:  # low abundancy
                if ct_delta > 0.40:
                    very_intersting_filters[idx] = append_string
                if ct_delta > 0.30:
                    intersting_filters[idx] = append_string
    return intersting_filters, very_intersting_filters


### MARKER SELECTED vs ALL
# gets me distribution in an ALL vs SELECTED cell fashion
# sort markers in decreasing order of KS statistic
def get_sorted_ks(arraylike_one, arraylike_two, labels):
    ks_values = []
    p_vals = []
    for j in range(len(labels)):
        ks = stats.ks_2samp(arraylike_one[:, j], arraylike_two[:, j])
        ks_values.append(ks[0])
        p_vals.append(ks[1])
    # sort markers in decreasing order of KS statistic
    sorted_idx = np.argsort(np.array(ks_values))[::-1]
    p_vals = [p_vals[i] for i in sorted_idx]
    sorted_labels = [labels[i] for i in sorted_idx]
    sorted_ks = [('KS = %.2f' % ks_values[i]) for i in sorted_idx]
    return sorted_ks, sorted_idx, sorted_labels, p_vals


def handle_directories(file, is_plot=False):
    plotdir = os.path.join(file[:-12], 'plots')
    csv_dir = os.path.join(file[:-12], 'selected_cells')
    filters_dir = f'{file[:-12]}/selected_cells/filters'
    abundancy_dir = f'{filters_dir}/abundancies'
    if is_plot == True:
        try:
            shutil.rmtree(plotdir)
            shutil.rmtree(csv_dir)
        except IOError as e:
            print("Error: %s" % (e.strerror))
            pass
    mkdir_p(csv_dir)
    mkdir_p(filters_dir)
    mkdir_p(abundancy_dir)
    return abundancy_dir, filters_dir, plotdir


def calc_frequencies(df, ref_dict, freq_col='cluster', return_list=False):
    frequency_dict = dict()
    for key in ref_dict.keys():  # these are my clusters
        cell_type_count = (df[df[freq_col] == key]).count()[0]  # how many cells of type key
        freq = str(cell_type_count / df.shape[0])  # freq
        frequency_dict[key] = float(freq)

    if return_list:
        return list(frequency_dict.values())
    return frequency_dict


def split_test_train_valid(*args, train_perc=0.8, test_perc=0.2, valid_perc=0.5, seed=123, savedir=None):
    results = dict()
    for i, value in enumerate(args):
        length = len(value)
        results[f'{i}_test'] = value[:int(test_perc * length)]
        results[f'{i}_train'] = np.asarray(value[int(test_perc * length):], dtype=object)
        border = int(valid_perc * len(results[f'{i}_test']))
        results[f'{i}_valid'] = np.asarray(results[f'{i}_test'][border:], dtype=object)
        results[f'{i}_test'] = np.asarray(results[f'{i}_test'][:border], dtype=object)
    if savedir is not None:
        with open(f'{savedir}/saved_data_splits.pkl', 'wb') as f:
            pickle.dump(results, f)
    return results.values()


def get_fitted_model(X_train, X_valid, y_train, y_valid,
                     nrun=15, ncell=200, nsubset=1000, nfilters=[3], maxpool_percentages=[0.01, 1., 5., 20., 100.],
                     coeff_l2=0.0001, coeff_l1=0, max_epochs=50, learning_rate=None,
                     outdir='default_output_dir', subset_selection='random',
                     per_sample=True, regression=True, verbose=True, result=False, mtl_tasks=1,
                     quant_normed=False, scale=True, loss_mode=None):
    model = CellCnn(nrun=nrun, ncell=ncell, nsubset=nsubset,
                    nfilter_choice=nfilters, learning_rate=learning_rate,
                    coeff_l2=coeff_l2, coeff_l1=coeff_l1, subset_selection=subset_selection,
                    max_epochs=max_epochs, per_sample=per_sample,
                    regression=regression, verbose=verbose, mtl_tasks=mtl_tasks,
                    maxpool_percentages=maxpool_percentages,
                    quant_normed=quant_normed, scale=scale, loss_mode=loss_mode)

    model = model.fit(train_samples=X_train, train_phenotypes=y_train,
                      valid_samples=X_valid, valid_phenotypes=y_valid,
                      outdir=outdir)
    return model


def create_input_data(rand_seed, train_perc=0.7, test_perc=0.3, outdir='default_outdir',
                      infile='data/input/cohort_denoised_clustered_diagnosis_patients.csv'):
    cytokines = ['CD69', 'CXCR5', 'CCR2', 'CD57', 'CD28', 'IL.9', 'IL.4', 'CD27', 'IL.3',
                 'IL.13', 'IL.2', 'CD103', 'PD.1', 'IL.10', 'CD20', 'CCR7', 'TNF.a', 'CD8', 'GM.CSF', 'IFN.g', 'CCR6',
                 'IL.22', 'CD45RO', 'TCRgd', 'CD56',
                 'IL.6', 'CD45RA', 'CD25', 'CD4', 'CD3', 'IL.21', 'IL.17A', 'CXCR4',
                 'CCR4', 'CD14']
    mkdir_p(outdir)
    cluster_to_celltype_dict = {0: 'b', 1: 'cd4', 3: 'nkt', 8: 'cd8', 10: 'nk', 11: 'my', 16: 'dg'}
    np.random.seed(rand_seed)
    df = pd.read_csv(infile, index_col=0)
    df = df.drop_duplicates()  ### reduces overfitting at cost of fewer data
    rrms_df = df[df['diagnosis'] == 'RRMS']
    rrms_patients2df = {id: patient_df.drop(columns=['diagnosis', 'gate_source']) for id, patient_df in
                        rrms_df.groupby('gate_source')}
    nindc_df = df[df['diagnosis'] == 'NINDC']
    nindc_patients2df = {id: patient_df.drop(columns=['diagnosis', 'gate_source']) for id, patient_df in
                         nindc_df.groupby('gate_source')}
    #### here we could see freq differences across the 2 groups
    print('Frequencies: ')
    rrms_patients_freq = {id: calc_frequencies(patient_df, cluster_to_celltype_dict, return_list=True) for
                          id, patient_df in
                          rrms_patients2df.items()}
    nindc_patients_freq = {id: calc_frequencies(patient_df, cluster_to_celltype_dict, return_list=True) for
                           id, patient_df
                           in nindc_patients2df.items()}
    print('DONE')
    # we got 31 patients each
    batch_size_dict = dict()
    ### desease states 1 = RRMS and 0 = NINDC
    selection_pool_rrms_cd8 = [(df.loc[:, df.columns != 'cluster'], rrms_patients_freq[patient], 1)
                               for patient, df in rrms_patients2df.items()]
    selection_pool_nindc_cd8 = [(df.loc[:, df.columns != 'cluster'], nindc_patients_freq[patient], 0)
                                for patient, df in nindc_patients2df.items()]

    # make sure list are equally long:
    if len(selection_pool_rrms_cd8) > len(selection_pool_nindc_cd8):
        selection_pool_rrms_cd8 = selection_pool_rrms_cd8[:len(selection_pool_nindc_cd8)]
    elif len(selection_pool_rrms_cd8) < len(selection_pool_nindc_cd8):
        selection_pool_nindc_cd8 = selection_pool_nindc_cd8[:len(selection_pool_rrms_cd8)]

    all_chunks = selection_pool_rrms_cd8 + selection_pool_nindc_cd8
    np.random.shuffle(all_chunks)  # to get differing phenotypes...

    idxs = [selection[0].index for selection in all_chunks]
    X = [selection[0].to_numpy() for selection in all_chunks]
    freqs = [selection[1] for selection in all_chunks]
    Y = [selection[2] for selection in all_chunks]

    X_test, X_train, X_valid, freq_test, freq_train, freq_valid, y_test, y_train, y_valid, idxs_test, idxs_train, idxs_valid = split_test_train_valid(
        X, freqs, Y, idxs,
        train_perc=train_perc,
        test_perc=test_perc,
        savedir=outdir,
        valid_perc=0.4)

    ## metadata
    concat = np.concatenate((idxs_test, idxs_train, idxs_valid))
    idxs = np.hstack(concat)
    df_test = df.reindex(idxs)
    # %%
    concat = np.concatenate((X_test, X_train, X_valid))
    x = np.vstack(concat)
    metadata_df = pd.DataFrame(x, index=idxs, columns=cytokines)
    metadata_df['cluster'] = df_test['cluster']
    metadata_df['diagnosis'] = df_test['diagnosis']
    metadata_df['gate_source'] = df_test['gate_source']
    pd.DataFrame.equals(metadata_df, df_test)
    metadata_df.to_csv(f'{outdir}/metadata_r4_2.csv')
    print(f'done creating input data to {outdir}')
