from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split

import cellCnn
from cellCnn.model import CellCnn
import numpy as np

def get_chunks_from_df(patient_df, freq_df, desease_state=0, cluster=1, batch_size=100):
    too_few_data = []
    selection_pool = []
    for patient, df in patient_df.items():
        cell_types = df[df['cluster'] == cluster]
        if len(cell_types) < batch_size:
            too_few_data.append(df)  # todo maybe merge together several "too few ones"
            continue

        selection_idx = np.random.choice(cell_types.index, batch_size)
        selection = cell_types.loc[selection_idx, cell_types.columns != 'cluster']  # to get 'cluster' out
        all_freqs = freq_df[patient]
        selection_pool.append((selection, all_freqs, desease_state))
    return selection_pool, too_few_data


def get_chunks(idxs, size):
    idxs_chunks = []
    for i in range(0, len(list(idxs)), size):
        idxs_chunks.append(idxs[i:i + size])
    return idxs_chunks


def calc_frequencies(df, ref_dict, freq_col='cluster', return_list = False):
    frequency_dict = dict()
    for key in ref_dict.keys(): # these are my clusters
        cell_type_count = (df[df[freq_col] == key]).count()[0] # how many cells of type key
        freq = str(cell_type_count / df.shape[0]) # freq
        frequency_dict[key] = float(freq)

    if return_list:
        return list(frequency_dict.values())
    return frequency_dict


# valid perc is the mount of the TEST set that is for validation
def split_test_valid_train(X, y, test_perc=0.8, train_perc=0.2, valid_perc=0.5, seed=123):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_perc,
                                                        train_size=train_perc,
                                                        random_state=seed)
    ### and one to get validation samples as well:
    X_valid, X_test, y_valid, y_test = train_test_split(X_test,
                                                        y_test,
                                                        test_size=valid_perc,
                                                        train_size=1 - valid_perc,
                                                        random_state=seed)
    return X_test, X_train, X_valid, y_test, y_train, y_valid


def get_fitted_model(X_train, X_valid, y_train, y_valid,
                     nrun=15, ncell=200, nsubset=1000, nfilters=[3],
                     coeff_l2=0.0001, coeff_l1=0, max_epochs=50, learning_rate=None,
                     outdir='default_output_dir',
                     per_sample=True, regression=True, verbose=True):
    ### for Pycharm i need to update the CellCNN model EVERY TIME I CHANGE stuff
    import importlib
    importlib.reload(cellCnn)
    #importlib.reload(cellCnn.model)
    #from cellCnn.model import CellCnn
    ## parameters from PBMC example
    ###per_sample bei regression

    mtl=False
    if any(isinstance(item, list) for item in y_train): ### is list of lists ?
        mtl=True

    model = CellCnn(nrun=nrun, ncell=ncell, nsubset=nsubset,
                    nfilter_choice=nfilters, learning_rate=learning_rate,
                    coeff_l2=coeff_l2, coeff_l1=coeff_l1,
                    max_epochs=max_epochs, per_sample=per_sample,
                    regression=regression, verbose=verbose)

    model.fit(train_samples=X_train, train_phenotypes=y_train,
              valid_samples=X_valid, valid_phenotypes=y_valid,
              outdir=outdir)
    return model




def print_regression_model_stats(test_pred, y_test):
    test_pred_list = [item for sublist in test_pred for item in sublist]
    print('\nModel predictions:\n', test_pred_list)
    print('\nTrue phenotypes:\n', y_test)
    print(f'RMSE: {mean_squared_error(test_pred, y_test)}')
    print(
        f'R2: {r2_score(test_pred, y_test)}')  # the higher the more variation is explained ” …the proportion of the variance in the dependent variable that is predictable from the independent variable(s).”
    # 1 - tot. sum residuals (ytest[i] – preds[i]) **2) /  tot. sum squares (ytest[i] - mean(ytest[i]))
    print(
        f'explained_variance_score: {explained_variance_score(test_pred, y_test)}')  # ” …the proportion of the variance in the dependent variable that is predictable from the independent variable(s).”
    return mean_squared_error(test_pred, y_test), r2_score(test_pred, y_test)