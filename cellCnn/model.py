""" Copyright 2016-2017 ETH Zurich, Eirini Arvaniti and Manfred Claassen.

This module contains functions for performing a CellCnn analysis.

"""

import sys
import os
import copy
import logging
from collections import OrderedDict

import keras.backend
import keras.utils
import numpy
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from cellCnn.utils import combine_samples, normalize_outliers_to_control, generate_subsets_mtl, combine_samples_mtl
from cellCnn.utils import cluster_profiles, keras_param_vector
from cellCnn.utils import generate_subsets, generate_biased_subsets
from cellCnn.utils import get_filters_classification, get_filters_regression
from cellCnn.utils import mkdir_p

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, regularizers, optimizers, callbacks, losses

from loss import RevisedUncertaintyLoss

logger = logging.getLogger(__name__)


class CellCnn(object):
    """ Creates a CellCnn model.

    Args:
        - ncell :
            Number of cells per multi-cell input.
        - nsubset :
            Total number of multi-cell inputs that will be generated per class, if
            `per_sample` = `False`. Total number of multi-cell inputs that will be generated from
            each input sample, if `per_sample` = `True`.
        - per_sample :
            Whether the `nsubset` argument refers to each class or each input sample.
            For regression problems, it is automatically set to `True`.
        - subset_selection :
            Can be 'random' or 'outlier'. Generate multi-cell inputs uniformly at
            random or biased towards outliers. The latter option is only relevant for detection of
            extremely rare (frequency < 0.1%) cell populations.
        - maxpool_percentages :
            A list specifying candidate percentages of cells that will be max-pooled per
            filter. For instance, mean pooling corresponds to `maxpool_percentages` = `[100]`.
        - nfilter_choice :
            A list specifying candidate numbers of filters for the neural network.
        - scale :
            Whether to z-transform each feature (mean = 0, standard deviation = 1) prior to
            training.
        - quant_normed :
            Whether the input samples have already been pre-processed with quantile
            normalization. In this case, each feature is zero-centered by subtracting 0.5.
        - nrun :
            Number of neural network configurations to try (should be set >= 3).
        - regression :
            Set to `True` for a regression problem. Default is `False`, which corresponds
            to a classification setting.
        - learning_rate :
            Learning rate for the Adam optimization algorithm. If set to `None`,
            learning rates in the range [0.001, 0.01] will be tried out.
        - dropout :
            Whether to use dropout (at each epoch, set a neuron to zero with probability
            `dropout_p`). The default behavior 'auto' uses dropout when `nfilter` > 5.
        - dropout_p :
            The dropout probability.
        - coeff_l1 :
            Coefficient for L1 weight regularization.
        - coeff_l2 :
            Coefficient for L2 weight regularization.
        - max_epochs :
            Maximum number of iterations through the data.
        - patience :
            Number of epochs before early stopping (stops if the validation loss does not
            decrease anymore).
        - dendrogram_cutoff :
            Cutoff for hierarchical clustering of filter weights. Clustering is
            performed using cosine similarity, so the cutof should be in [0, 1]. A lower cutoff will
            generate more clusters.
        - accur_thres :
            Keep filters from models achieving at least this accuracy. If less than 3
            models pass the accuracy threshold, keep filters from the best 3 models.
    """

    def __init__(self, ncell=200, nsubset=1000, per_sample=False, subset_selection='random',
                 maxpool_percentages=None, scale=True, quant_normed=False,
                 nfilter_choice=None, dropout='auto', dropout_p=.5,
                 coeff_l1=0, coeff_l2=0.0001, learning_rate=None,
                 regression=False, max_epochs=20, patience=5, nrun=15, dendrogram_cutoff=0.4,
                 accur_thres=.95, verbose=1, mtl_tasks=1):

        # initialize model attributes
        self.scale = scale
        self.quant_normed = quant_normed
        self.nrun = nrun
        self.regression = regression
        self.ncell = ncell
        self.nsubset = nsubset
        self.per_sample = per_sample
        self.subset_selection = subset_selection
        self.maxpool_percentages = maxpool_percentages
        self.nfilter_choice = nfilter_choice
        self.learning_rate = learning_rate
        self.coeff_l1 = coeff_l1
        self.coeff_l2 = coeff_l2
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.max_epochs = max_epochs
        self.patience = patience
        self.dendrogram_cutoff = dendrogram_cutoff
        self.accur_thres = accur_thres
        self.verbose = verbose
        self.results = None
        self.mtl_tasks = mtl_tasks

    def fit(self, train_samples, train_phenotypes, outdir, valid_samples=None,
            valid_phenotypes=None, generate_valid_set=True):

        """ Trains a CellCnn model.

        Args:
            - train_samples :
                List with input samples (e.g. cytometry samples) as numpy arrays.
            - train_phenotypes :
                List of phenotypes associated with the samples in `train_samples`.
            - outdir :
                Directory where output will be generated.
            - valid_samples :
                List with samples to be used as validation set while training the network.
            - valid_phenotypes :
                List of phenotypes associated with the samples in `valid_samples`.
            - generate_valid_set :
                If `valid_samples` is not provided, generate a validation set
                from the `train_samples`.

        Returns:
            A trained CellCnn model with the additional attribute `results`. The attribute `results`
            is a dictionary with the following entries:

            - clustering_result : clustered filter weights from all runs achieving \
                validation accuracy above the specified threshold `accur_thres`
            - selected_filters : a consensus filter matrix from the above clustering result
            - best_3_nets : the 3 best models (achieving highest validation accuracy)
            - best_net : the best model
            - w_best_net : filter and output weights of the best model
            - accuracies : list of validation accuracies achieved by different models
            - best_model_index : list index of the best model
            - config : list of neural network configurations used
            - scaler : a z-transform scaler object fitted to the training data
            - n_classes : number of output classes
        """

        res = train_model(train_samples, train_phenotypes, outdir,
                          valid_samples, valid_phenotypes, generate_valid_set,
                          scale=self.scale, nrun=self.nrun, regression=self.regression,
                          ncell=self.ncell, nsubset=self.nsubset, per_sample=self.per_sample,
                          subset_selection=self.subset_selection,
                          maxpool_percentages=self.maxpool_percentages,
                          nfilter_choice=self.nfilter_choice,
                          learning_rate=self.learning_rate,
                          coeff_l1=self.coeff_l1, coeff_l2=self.coeff_l2,
                          dropout=self.dropout, dropout_p=self.dropout_p,
                          max_epochs=self.max_epochs,
                          patience=self.patience, dendrogram_cutoff=self.dendrogram_cutoff,
                          accur_thres=self.accur_thres, verbose=self.verbose, mtl_tasks=self.mtl_tasks)
        self.results = res
        return self

    def predict(self, new_samples, ncell_per_sample=None):
        """ Makes predictions for new samples.
            Takes the mean of the best 3 nets´ predictions and return it for

        Args:
            - new_samples :
                List with input samples (numpy arrays) for which predictions will be made.
            - ncell_per_sample :
                Size of the multi-cell inputs (only one multi-cell input is created
                per input sample). If set to None, the size of the multi-cell inputs equals the
                minimum size in `new_samples`.

        Returns:
            y_pred : Phenotype predictions for `new_samples`. OR for mtl_tasks > 1 predictions for all multi-task
             learning tasks in the order that labels have been given in
        """
        if ncell_per_sample is None:
            ncell_per_sample = np.min([x.shape[0] for x in new_samples])
        logger.info(f"Predictions based on multi-cell inputs containing {ncell_per_sample} cells.")

        # z-transform the new samples if we did that for the training samples
        scaler = self.results['scaler']
        if scaler is not None:
            new_samples = copy.deepcopy(new_samples)
            new_samples = [scaler.transform(x) for x in new_samples]

        nmark = new_samples[0].shape[1]
        n_classes = self.results['n_classes']

        # get the configuration of the top 3 models
        accuracies = self.results['accuracies']
        sorted_idx = np.argsort(accuracies)[::-1][:3]
        config = self.results['config']

        # y_pred = np.zeros((3, len(new_samples), n_classes))
        y_pred = dict()
        for i_enum, i in enumerate(sorted_idx):
            nfilter = config['nfilter'][i]
            maxpool_percentage = config['maxpool_percentage'][i]
            ncell_pooled = max(1, int(maxpool_percentage / 100. * ncell_per_sample))

            # build the model architecture
            model = build_model(ncell_per_sample, nmark,
                                nfilter=nfilter, coeff_l1=0, coeff_l2=0,
                                k=ncell_pooled, dropout=False, dropout_p=0,
                                regression=self.regression, n_classes=n_classes, lr=0.01, mtl_tasks=self.mtl_tasks)

            # and load the learned filter and output weights
            weights = self.results['best_3_nets'][i_enum]
            model.set_weights(weights)
            #        weights: a list of Numpy arrays. The number of arrays and their shape
            # must match number of the dimensions of the weights of the optimizer
            # (i.e. it should match the output of `get_weights`).

            # select a random subset of `ncell_per_sample` and make predictions
            new_samples = [shuffle(x)[:ncell_per_sample].reshape(1, ncell_per_sample, nmark)
                           for x in new_samples]
            data_test = np.vstack(new_samples)
            prediction = model.predict(data_test)
            y_pred[i_enum] = prediction

        if self.mtl_tasks == 1:
            result = np.mean(y_pred, axis=0)
        else:
            result = []
            for task_nr in range(self.mtl_tasks):
                to_mean = []
                for idx, pred in enumerate(y_pred.values()):
                    to_mean.append(pred[task_nr])
                mean = np.mean(np.array(to_mean), axis=0)
                result.append(mean)
        return result


# todo make position of classification labels more dynamic
def train_model(train_samples, train_phenotypes, outdir,
                valid_samples=None, valid_phenotypes=None, generate_valid_set=True,
                scale=True, quant_normed=False, nrun=20, regression=False,
                ncell=200, nsubset=1000, per_sample=False, subset_selection='random',
                maxpool_percentages=None, nfilter_choice=None,
                learning_rate=None, coeff_l1=0, coeff_l2=1e-4, dropout='auto', dropout_p=.5,
                max_epochs=20, patience=5,
                dendrogram_cutoff=0.4, accur_thres=.95, verbose=1, mtl_tasks=1):
    """ Performs a CellCnn analysis """

    train_phenotype_dict = OrderedDict()
    valid_phenotype_dict = OrderedDict()
    if any(isinstance(item, np.ndarray) for item in train_phenotypes) or \
            any(isinstance(item, list) for item in train_phenotypes):  ### is list of lists ?
        if mtl_tasks != len(train_phenotypes):
            sys.stderr.write('"mtl_tasks" must be the length of the "train_phenotypes" list.\n')

        for task in range(mtl_tasks):
            train_phenotype_dict[task] = np.asarray(train_phenotypes[task])
    else:
        train_phenotype_dict[0] = np.asarray(train_phenotypes)

    ### if you pass in valid train valid phenos should be given as well
    if any(isinstance(item, np.ndarray) for item in valid_phenotypes) or any(
            isinstance(item, list) for item in valid_phenotypes):  ### is list of lists ?
        for task in range(mtl_tasks):
            valid_phenotype_dict[task] = np.asarray(valid_phenotypes[task])
    else:
        if valid_samples is not None:
            valid_phenotype_dict[0] = valid_phenotypes

    if maxpool_percentages is None:
        maxpool_percentages = [0.01, 1., 5., 20., 100.]
    if nfilter_choice is None:
        nfilter_choice = list(range(3, 10))

    mkdir_p(outdir)

    if nrun < 3:
        logger.info(f"The nrun argument should be >= 3, setting it to 3.")
        nrun = 3

    # copy the list of samples so that they are not modified in place
    train_samples = copy.deepcopy(train_samples)
    if valid_samples is not None:
        valid_samples = copy.deepcopy(valid_samples)

    # normalize extreme values
    # we assume that 0 corresponds to the control class
    if subset_selection == 'outlier':
        ctrl_list = [train_samples[i] for i in np.where(np.array(train_phenotype_dict[0]) == 0)[0]]
        test_list = [train_samples[i] for i in np.where(np.array(train_phenotype_dict[0]) != 0)[0]]
        train_samples = normalize_outliers_to_control(ctrl_list, test_list)

        if valid_samples is not None:
            ctrl_list = [valid_samples[i] for i in np.where(np.array(valid_phenotype_dict[0]) == 0)[0]]
            test_list = [valid_samples[i] for i in np.where(np.array(valid_phenotype_dict[0]) != 0)[0]]
            valid_samples = normalize_outliers_to_control(ctrl_list, test_list)

    # merge all input samples (X_train, X_valid)
    # and generate an identifier for each of them (train_id, valid_id)
    train_sample_ids = np.arange(len(train_phenotype_dict[0]))  ## refers to the sample, not to the unique labels

    if (valid_samples is None) and (not generate_valid_set):
        X_train, id_train = combine_samples(train_samples, train_sample_ids)

    elif (valid_samples is None) and generate_valid_set:
        X, sample_id = combine_samples(train_samples, train_sample_ids)

        for task in range(mtl_tasks):
            valid_phenotype_dict[task] = train_phenotype_dict[task]

        # split into train-validation partitions
        eval_folds = 5
        kf = StratifiedKFold(n_splits=eval_folds)
        train_indices, valid_indices = next(kf.split(X, sample_id))
        X_train, id_train = X[train_indices], sample_id[train_indices]
        X_valid, id_valid = X[valid_indices], sample_id[valid_indices]

    else:  # if valid samples is NOT None
        X_train, id_train = combine_samples(train_samples, train_sample_ids)
        valid_sample_ids = np.arange(len(valid_phenotype_dict[0]))
        X_valid, id_valid = combine_samples(valid_samples, valid_sample_ids)

    if quant_normed:
        z_scaler = StandardScaler(with_mean=True, with_std=False)
        z_scaler.fit(0.5 * np.ones((1, X_train.shape[1])))
        X_train = z_scaler.transform(X_train)
    elif scale:
        z_scaler = StandardScaler(with_mean=True, with_std=True)
        z_scaler.fit(X_train)
        X_train = z_scaler.transform(X_train)
    else:
        z_scaler = None

    X_train, id_train = shuffle(X_train, id_train)

    # an array containing the phenotype for each single cell
    y_train = train_phenotype_dict[0][id_train]

    if (valid_samples is not None) or generate_valid_set:
        if scale:
            X_valid = z_scaler.transform(X_valid)

        X_valid, id_valid = shuffle(X_valid, id_valid)
        y_valid = valid_phenotype_dict[0][id_valid]

    # number of measured markers
    nmark = X_train.shape[1]

    # generate multi-cell inputs
    logger.info("Generating multi-cell inputs...")

    # todo mtl FIND OUT if i need to  do this only for phenotype or as well for other tasks
    if subset_selection == 'outlier':
        # here we assume that class 0 is always the control class
        x_ctrl_train = X_train[y_train == 0]
        to_keep = int(0.1 * (X_train.shape[0] / len(train_phenotype_dict[0])))
        nsubset_ctrl = nsubset // np.sum(train_phenotype_dict[0] == 0)

        # generate a fixed number of subsets per class
        nsubset_biased = [0]
        for pheno in range(1, len(np.unique(train_phenotype_dict[0]))):
            nsubset_biased.append(nsubset // np.sum(train_phenotype_dict[0] == pheno))

        X_tr, y_tr = generate_biased_subsets(X_train, train_phenotype_dict[0], id_train, x_ctrl_train,
                                             nsubset_ctrl, nsubset_biased, ncell, to_keep,
                                             id_ctrl=np.where(train_phenotype_dict[0] == 0)[0],
                                             id_biased=np.where(train_phenotype_dict[0] != 0)[0])
        # save those because it takes long to generate
        # np.save(os.path.join(outdir, 'X_tr.npy'), X_tr)
        # np.save(os.path.join(outdir, 'y_tr.npy'), y_tr)
        # X_tr = np.load(os.path.join(outdir, 'X_tr.npy'))
        # y_tr = np.load(os.path.join(outdir, 'y_tr.npy'))

        if (valid_samples is not None) or generate_valid_set:
            x_ctrl_valid = X_valid[y_valid == 0]
            nsubset_ctrl = nsubset // np.sum(valid_phenotype_dict[0] == 0)

            # generate a fixed number of subsets per class
            nsubset_biased = [0]
            for pheno in range(1, len(np.unique(valid_phenotype_dict[0]))):
                nsubset_biased.append(nsubset // np.sum(valid_phenotype_dict[0] == pheno))

            to_keep = int(0.1 * (X_valid.shape[0] / len(valid_phenotype_dict[0])))
            X_v, y_v = generate_biased_subsets(X_valid, valid_phenotype_dict[0], id_valid, x_ctrl_valid,
                                               nsubset_ctrl, nsubset_biased, ncell, to_keep,
                                               id_ctrl=np.where(valid_phenotype_dict[0] == 0)[0],
                                               id_biased=np.where(valid_phenotype_dict[0] != 0)[0])
            # save those because it takes long to generate
            # np.save(os.path.join(outdir, 'X_v.npy'), X_v)
            # np.save(os.path.join(outdir, 'y_v.npy'), y_v)
            # X_v = np.load(os.path.join(outdir, 'X_v.npy'))
            # y_v = np.load(os.path.join(outdir, 'y_v.npy'))
        else:
            cut = X_tr.shape[0] // 5
            X_v = X_tr[:cut]
            y_v = y_tr[:cut]
            X_tr = X_tr[cut:]
            y_tr = y_tr[cut:]
    else:
        # generate 'nsubset' multi-cell inputs per input sample
        if per_sample:  # for regression
            y_inputs_train = train_phenotype_dict.values()
            X_tr, y_tr = generate_subsets_mtl(X_train, y_inputs_train, id_train,
                                              nsubset, ncell, per_sample)

            if (valid_samples is not None) or generate_valid_set:
                y_inputs_valid = valid_phenotype_dict.values()
                X_v, y_v = generate_subsets_mtl(X_valid, y_inputs_valid, id_valid,
                                                nsubset, ncell, per_sample)

        # generate 'nsubset' multi-cell inputs per class
        else:
            nsubset_list = []
            for pheno in range(len(np.unique(train_phenotype_dict[0]))):
                nsubset_list.append(nsubset // np.sum(
                    train_phenotype_dict[0] == pheno))  # out of this we will pick the amount of subsets
            X_tr, y_tr = generate_subsets_mtl(X_train, train_phenotype_dict[0], id_train,
                                              nsubset_list, ncell, per_sample)

            # todo nsubset_list[0] to get the values of the list so i get an array in return of a certain shape:
            # check if listr elements usually differ from each other (if so we got a label imbalance here!). As well i play around with per_sample,
            for task_nr in range(1, mtl_tasks):  # here i basically add the regression tasks to y_tr as well
                input_subsets = int(len(X_tr) / len(train_phenotype_dict[0]))
                _, y_tr_task = generate_subsets_mtl(X_train, train_phenotype_dict[task_nr], id_train,
                                                    nsubsets=input_subsets, ncell=ncell, per_sample=True)
                y_tr.append(y_tr_task[0])

            if (valid_samples is not None) or generate_valid_set:
                nsubset_list = []
                for pheno in range(len(np.unique(valid_phenotype_dict[0]))):
                    nsubset_list.append(nsubset // np.sum(valid_phenotype_dict[0] == pheno))
                X_v, y_v = generate_subsets_mtl(X_valid, valid_phenotype_dict[0], id_valid,
                                                nsubset_list, ncell, per_sample)

                for task_nr in range(1, mtl_tasks):  # here i basically add the regression tasks to y_tr as well
                    input_subsets = int(len(X_v) / len(valid_phenotype_dict[0]))
                    _, y_v_task = generate_subsets_mtl(X_train, valid_phenotype_dict[task_nr], id_valid,
                                                       nsubsets=input_subsets, ncell=ncell, per_sample=True)
                    y_v.append(y_v_task[0])
    logger.info("Done.")

    # neural network configuration
    # batch size
    bs = 200

    # keras needs (nbatch, ncell, nmark)
    X_tr = np.swapaxes(X_tr, 2, 1)
    X_v = np.swapaxes(X_v, 2, 1)
    n_classes = 1

    if not regression:
        n_classes = len(np.unique(train_phenotype_dict[0]))  # since the first input need to be the phenotype
        y_tr[0] = keras.utils.to_categorical(y_tr[0], n_classes)
        y_v[0] = keras.utils.to_categorical(y_v[0], n_classes)

    # train some neural networks with different parameter configurations
    accuracies = np.zeros(nrun)
    frequency_loss = dict()
    w_store = dict()
    config = dict()
    config['nfilter'] = []
    config['learning_rate'] = []
    config['maxpool_percentage'] = []
    lr = learning_rate

    for irun in range(nrun):
        if verbose:
            logger.info(f"Training network: {irun + 1}")
        if learning_rate is None:
            lr = 10 ** np.random.uniform(-3, -2)
            config['learning_rate'].append(lr)

        # choose number of filters for this run
        nfilter = np.random.choice(nfilter_choice)
        config['nfilter'].append(nfilter)
        logger.info(f"Number of filters: {nfilter}")

        # choose number of cells pooled for this run
        mp = maxpool_percentages[irun % len(maxpool_percentages)]
        config['maxpool_percentage'].append(mp)
        k = max(1, int(mp / 100. * ncell))
        logger.info(f"Cells pooled: {k}")

        # build the neural network
        model = build_model(ncell, nmark, nfilter,
                            coeff_l1, coeff_l2, k,
                            dropout, dropout_p, regression, n_classes, lr, mtl_tasks=mtl_tasks)
        statspath= os.path.join(outdir, 'stats')
        keras.utils.plot_model(model, statspath, show_shapes=True)

        print(model.summary())
        # print('\nlosses:', model.losses())

        filepath = os.path.join(outdir, 'nnet_run_%d.hdf5' % irun)
        try:
            check = callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='auto')
            earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='auto')

            y_trains = [y_tr[task] for task in range(mtl_tasks)]
            y_valids = [y_v[task] for task in range(mtl_tasks)]
            print('Trainfit incoming')
            hist = model.fit([X_tr, *y_trains], y=y_trains,
                      epochs=max_epochs, batch_size=bs, callbacks=[check, earlyStopping],
                      validation_data=([X_v, *y_valids], y_valids), verbose=verbose)
            print('Trainfit done')

            # load the model from the epoch with highest validation accuracy
            model.load_weights(filepath)

            if not regression:
                valid_metric_results = dict()
                valid_metric = model.evaluate([X_v, *y_valids], [y_v[task] for task in range(mtl_tasks)])[-1]
                if not isinstance(valid_metric, list):
                    valid_metric_results[0] = valid_metric

                logger.info(f"Best validation accuracy: {valid_metric_results[0]:.2f}")
                accuracies[irun] = valid_metric_results[0]

                # take the one for phenotype task

                for task in range(1, mtl_tasks):
                    logger.info(f"out {task} loss: {valid_metric_results[0]}")

            else:
                train_metric = model.evaluate([X_tr, *y_trains], [y_tr[task] for task in range(mtl_tasks)],
                                              batch_size=bs)
                valid_metric = model.evaluate([X_v, *y_valids], [y_v[task] for task in range(mtl_tasks)], batch_size=bs)

                if mtl_tasks == 1:
                    logger.info(f"Best train loss: {train_metric}")
                    logger.info(f"Best valid loss: {valid_metric}")
                    accuracies[irun] = - valid_metric
                else:
                    logger.info(f"Best train tot. loss: {train_metric[0]}")
                    logger.info(f"Best valid tot. loss: {valid_metric[0]}")
                    accuracies[irun] = - valid_metric[0]

                    freq_metrics = []
                    for task in range(1, mtl_tasks):
                        logger.info(f"out train {task} loss: {train_metric[task]}")
                        logger.info(f"out valid {task} loss: {valid_metric[task]}")
                        freq_metrics.append(- valid_metric[task])
                    frequency_loss[irun] = freq_metrics

            # extract the network parameters
            w_store[irun] = model.get_weights()

        except Exception as e:
            sys.stderr.write('An exception was raised during training the network.\n')
            sys.stderr.write(str(e) + '\n')

    # the top 3 performing networks
    ### todo maybe embed the freq performance as well ???
    model_sorted_idx = np.argsort(accuracies)[::-1][:3]
    best_3_nets = [w_store[i] for i in model_sorted_idx]
    best_net = best_3_nets[0]
    best_accuracy_idx = model_sorted_idx[0]

    # weights from the best-performing network
    w_best_net = keras_param_vector(best_net)

    # post-process the learned filters
    # cluster weights from all networks that achieved accuracy above the specified thershold
    w_cons, cluster_res = cluster_profiles(w_store, nmark, accuracies, accur_thres,
                                           dendrogram_cutoff=dendrogram_cutoff)
    results = {
        'clustering_result': cluster_res,
        'selected_filters': w_cons,
        'best_net': best_net,
        'best_3_nets': best_3_nets,
        'w_best_net': w_best_net,
        'accuracies': accuracies,
        'best_model_index': best_accuracy_idx,
        'config': config,
        'scaler': z_scaler,
        'n_classes': n_classes
    }

    if (valid_samples is not None) and (w_cons is not None):
        maxpool_percentage = config['maxpool_percentage'][best_accuracy_idx]
        if regression:
            for task in range(mtl_tasks):
                tau = get_filters_regression(w_cons, z_scaler, valid_samples, valid_phenotype_dict[task],
                                             maxpool_percentage)

                if task == 0:  # this is mainly here for keeping the "old" output the same (STL way)
                    results['filter_tau'] = tau
                else:
                    results[f'filter_tau_{task}'] = tau

        else:
            for task in range(mtl_tasks):
                filter_diff = get_filters_classification(w_cons, z_scaler, valid_samples,
                                                         valid_phenotype_dict[task], maxpool_percentage)

                if task == 0:  # this is mainly here for keeping the "old" output the same (STL way)
                    results['filter_diff'] = filter_diff
                else:
                    results[f'filter_diff_{task}'] = filter_diff
    return results


# loss list contains loss functions
# !!trick was that this return a loss FUNCTION and not a value! therefore i fucked up a lot of time...
def uncertainty_loss(losses, model):
    # kendall: Sigma is the noise parameter! As it increases, the loss decreases
    print('Custom loss fn entered')
    sigma_squares = []
    for i, loss in enumerate(losses):
        sigma = tf.keras.backend.variable(value=0.5, dtype=tf.float32,
                                          name=f'Sigma_sq_{i}',
                                          constraint=lambda t: tf.clip_by_value(t, 0, 1))
        print(sigma)
        model.add_weight(sigma)
        sigma_squares.append(sigma)

    def get_loss(y_true, y_pred):
        print('get_loss entered, printing inputs')
        print(y_true)
        print(y_pred)
        print('SIGMAS')
        # print(tf.keras.backend.print_tensor(sigma_squares[0]))
        # print(tf.keras.backend.print_tensor(sigma_squares[1]))
        print('\n')

        loss = 0.
        for i, task_loss in enumerate(losses):
            factor = tf.math.divide_no_nan(1.0, tf.multiply(2.0, sigma_squares[i]))
            listed_loss = task_loss(y_true[i], y_pred[i])
            product = tf.multiply(factor, listed_loss)
            # tf.add(listed_loss, tf.math.log(sigma_squares[i]))
            loss = tf.add(loss, tf.add(product, tf.math.log(sigma_squares[i])))
        return loss

    return get_loss


def build_model(ncell, nmark, nfilter, coeff_l1, coeff_l2,
                k, dropout, dropout_p, regression, n_classes, lr=0.01, mtl_tasks=1):
    """ Builds the neural network architecture """

    # the input layer
    data_input = keras.Input(shape=(ncell, nmark), name='data_input')

    # the filters
    conv = layers.Conv1D(filters=nfilter,
                         kernel_size=1,
                         activation='relu',
                         kernel_initializer=initializers.RandomUniform(),
                         kernel_regularizer=regularizers.l1_l2(l1=coeff_l1, l2=coeff_l2),
                         name='conv1')(data_input)

    # the cell grouping part (top-k pooling)
    pooled = layers.Lambda(pool_top_k, output_shape=(nfilter,), arguments={'k': k})(conv)

    # possibly add dropout
    if dropout or ((dropout == 'auto') and (nfilter > 5)):
        pooled = layers.Dropout(rate=dropout_p)(pooled)

        ### todo modify loss functions here
    losses = []
    metrics = []
    if not regression:
        losses.append(tf.keras.losses.CategoricalCrossentropy())  # for phenotype
        metrics.append('accuracy')
        for task in range(1, mtl_tasks):
            losses.append(tf.keras.losses.MeanSquaredError())
            metrics.append('mean_squared_error')
    else:
        for task in range(mtl_tasks):
            losses.append(tf.keras.losses.MeanSquaredError())
    # todo  rename this to loss layer or sth. ...
    # check where i need to put that layer in
    # shaped_input = layers.Lambda(pool_top_k, output_shape=(nfilter,), arguments={'k': k})(data_input)
    ########################################################
    # -> erkenntniss i9ch will hier gar keine inputs haben sondern pro task den output!
    # pooled = RevisedUncertaintyLoss(losses)([])
    # evtld och als output ?

    # network prediction output
    output_layers = []
    if not regression:
        output = layers.Dense(units=n_classes,
                              activation='softmax',
                              kernel_initializer=initializers.RandomUniform(),
                              kernel_regularizer=regularizers.l1_l2(l1=coeff_l1, l2=coeff_l2),
                              name='output_desease')(pooled)
        y_pheno = layers.Input(shape=(2,), name='input_desease')
    else:
        output = layers.Dense(units=1,
                              activation='linear',
                              kernel_initializer=initializers.RandomUniform(),
                              kernel_regularizer=regularizers.l1_l2(l1=coeff_l1, l2=coeff_l2),
                              name='output_desease')(pooled)
        y_pheno = layers.Input(shape=(1,), name='input_desease')

    output_layers.append(output)

    ### this is the dummy for the freq regression for class this must be softmax as activation
    # all my atl tasks will be of regression type, start from 1 to not add the phenotype task twice
    # todo make it not mandadorty to put pheno first
    y_task_inputs = dict()
    for task in range(1, mtl_tasks):
        layer = layers.Dense(units=1,
                             activation='linear',
                             kernel_initializer=initializers.RandomUniform(),
                             kernel_regularizer=regularizers.l1_l2(l1=coeff_l1, l2=coeff_l2),
                             name=f'output_freq_{task}')(pooled)
        output_layers.append(layer)
        y_task_inputs[f'task_{task}'] = layers.Input(shape=(1,), name=f'input_task_{task}')

    # dynamically defining the inputs, the user needs to insert as many as tasks (obviously...)
    # todo is there any solution that solves this better (by taking the y_train´´ as those )
    out = RevisedUncertaintyLoss(loss_list=losses )([y_pheno, *y_task_inputs.values(), *output_layers])
    # evtl pooled statt output layers ?
    model = keras.Model(inputs=[data_input, y_pheno, *y_task_inputs.values()], outputs=out)

    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=None,
                  metrics=metrics)
    # todo  assert stuff

    return model


def pool_top_k(x, k):
    return tf.reduce_mean(tf.sort(x, axis=1, direction='DESCENDING')[:, :k, :], axis=1)
