import shutil
import pickle

from scipy import stats

from cellCnn.model import *


### MARKER SELECTED vs ALL
# gets me distribution in an ALL vs SELECTED cell fashion
# sort markers in decreasing order of KS statistic
def get_sorted_ks(arraylike_one, arraylike_two, labels):
    ks_values = []
    for j in range(len(labels)):
        ks = stats.ks_2samp(arraylike_one[:, j], arraylike_two[:, j])
        ks_values.append(ks[0])
    # sort markers in decreasing order of KS statistic
    sorted_idx = np.argsort(np.array(ks_values))[::-1]
    sorted_labels = [labels[i] for i in sorted_idx]
    sorted_ks = [('KS = %.2f' % ks_values[i]) for i in sorted_idx]
    return sorted_ks, sorted_idx, sorted_labels

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
        results[f'{i}_valid'] = np.asarray(results[f'{i}_test'][int(valid_perc * len(results[f'{i}_test'])):], dtype=object)
        results[f'{i}_test'] = np.asarray(results[f'{i}_test'][:int(valid_perc * len(results[f'{i}_test']))], dtype=object)
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

