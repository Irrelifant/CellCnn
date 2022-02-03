from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split

import cellCnn
from cellCnn.model import CellCnn


def get_chunks(idxs, size):
    idxs_chunks = []
    for i in range(0, len(list(idxs)), size):
        idxs_chunks.append(idxs[i:i + size])
    return idxs_chunks


def calc_frequencies(df, ref_dict, freq_col='cluster'):
    frequency_dict = dict()
    for key in ref_dict.keys():
        cell_type_count = (df[df[freq_col] == key]).count()[0]
        freq = str(cell_type_count / df.shape[0])
        print(f'For {key} we got a freq. {freq}')
        frequency_dict[key] = float(freq)
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
                     nrun=15, ncell=200, nsubset=500, nfilters=[3],
                     coeff_l2=1e-5, coeff_l1=0, max_epochs=50, learning_rate=10**2.5,
                     outdir='default_output_dir',
                     per_sample=True, regression=True):
    ### for Pycharm i need to update the CellCNN model EVERY TIME I CHANGE stuff
    import importlib
    importlib.reload(cellCnn)
    #importlib.reload(cellCnn.model)
    #from cellCnn.model import CellCnn
    ## parameters from PBMC example
    ###per_sample bei regression

    model = CellCnn(nrun=nrun, ncell=ncell, nsubset=nsubset, nfilter_choice=nfilters, learning_rate=learning_rate,
                    coeff_l2=coeff_l2, coeff_l1=coeff_l1, max_epochs=max_epochs, per_sample=per_sample,
                    regression=regression)

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
