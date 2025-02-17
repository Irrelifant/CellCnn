""" Copyright 2016-2017 ETH Zurich, Eirini Arvaniti and Manfred Claassen.

This module contains functions for performing a CellCnn analysis.

"""

import copy
import logging
import os
import sys
from collections import OrderedDict
from time import time

import keras.backend
import keras.utils
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import CSVLogger
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow import keras
from keras import layers, initializers, regularizers, callbacks
from tensorflow.python.keras.callbacks import TensorBoard
from cellCnn.loss_history import LossHistory
from cellCnn.losses.revised_uncertainty_loss import RevisedUncertaintyLossV2
from cellCnn.losses.uncertainty_loss import UncertaintyMultiLossLayer
from cellCnn.plotting import plot_model_losses
from cellCnn.utils import cluster_profiles, keras_param_vector
from cellCnn.utils import combine_samples, normalize_outliers_to_control, generate_subsets_mtl
from cellCnn.utils import generate_biased_subsets_mtl
from cellCnn.utils import get_filters_classification, get_filters_regression
from cellCnn.utils import mkdir_p

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
        - mtl_tasks :
            The amount of tasks that will be performed by the CellCNN model. The first task is currently always the phenotype related one, whilst following tasks will always be considered as regression tasks
        - loss_mode :
            The loss mode that will be followed. Only important in Multi-task learning.
            Currently toggleable between None, that will just take the summed average of the individual task losses,
            'uncertainty' that will add an additional loss layer to calculate the losses in Kendall´s uncertainty fashion
             and 'revised_uncertainty' that will add an additional loss layer to calculate the losses in a revised fashing of Kendall´s uncertainty.
             For the last two mentioned loss, distributions there is a '_v2' version that will take more loss layers (the Dense layers from the individual tasks) into account
    """

    def __init__(self, ncell=200, nsubset=1000, per_sample=False, subset_selection='random',
                 maxpool_percentages=None, scale=True, quant_normed=False,
                 nfilter_choice=None, dropout='auto', dropout_p=.5,
                 coeff_l1=0, coeff_l2=0.0001, learning_rate=None,
                 regression=False, max_epochs=20, patience=5, nrun=15, dendrogram_cutoff=0.4,
                 accur_thres=.95, verbose=1, mtl_tasks=1, loss_mode=None):

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
        self.loss_mode = loss_mode

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
                          accur_thres=self.accur_thres, verbose=self.verbose,
                          mtl_tasks=self.mtl_tasks, loss_mode=self.loss_mode)
        self.results = res
        return self

    def predict(self, new_samples, ncell_per_sample=None, mtl_inputs=None):
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
                                regression=self.regression, n_classes=n_classes,
                                lr=0.01, mtl_tasks=self.mtl_tasks, loss_mode=self.loss_mode)

            # and load the learned filter and output weights
            weights = self.results['best_3_nets'][i_enum]
            model.set_weights(weights)
            #        weights: a list of Numpy arrays. The number of arrays and their shape
            # must match number of the dimensions of the weights of the optimizer
            # (i.e. it should match the output of `get_weights`).

            # select a random subset of `ncell_per_sample` and make predictions
            new_samples = [shuffle(x)[:ncell_per_sample].reshape(1, ncell_per_sample, nmark)
                           for x in new_samples]
            data_converted = np.vstack(new_samples)

            ## This is there to solve numpy-array 'conversion to tensor' error
            data_converted = np.asarray([np.asarray(x).astype(np.float64) for x in data_converted])

            # since i still need at least one input if i got my mtl model
            if len(mtl_inputs) > 1:
                mtl_inputs_feed = []
                for i, task in enumerate(mtl_inputs):
                    if i == 0 and not self.regression:
                        # classification
                        nclasses = len(np.unique(task))
                        mtl_inputs_feed.append(tf.keras.utils.to_categorical(task, nclasses))
                    else:
                        # if i != 0 its definately a regression task
                        mtl_inputs_feed.append(np.asarray(task).astype(np.float64))

                prediction = model.predict([data_converted, *mtl_inputs_feed])
            else:
                prediction = model.predict(data_converted)

            y_pred[i_enum] = prediction

        if self.mtl_tasks == 1:
            # default mode
            results_as_np = np.array(list(y_pred.values()))
            result = np.mean(results_as_np, axis=0)
        else:
            # MTL-mode: take the mean of the prediction of every task
            result = []
            for task_nr in range(self.mtl_tasks):
                to_mean = [pred[task_nr] for pred in y_pred.values()]
                mean = np.mean(np.array(to_mean), axis=0)
                result.append(mean)
        return result


def train_model(train_samples, train_phenotypes, outdir,
                valid_samples=None, valid_phenotypes=None, generate_valid_set=True,
                scale=True, quant_normed=False, nrun=20, regression=False,
                ncell=200, nsubset=1000, per_sample=False, subset_selection='random',
                maxpool_percentages=None, nfilter_choice=None,
                learning_rate=None, coeff_l1=0, coeff_l2=1e-4, dropout='auto', dropout_p=.5,
                max_epochs=20, patience=5,
                dendrogram_cutoff=0.4, accur_thres=.95, verbose=1, mtl_tasks=1, loss_mode=None):
    """ Performs a CellCnn analysis """
    train_phenotype_dict = init_phenotype_dict(mtl_tasks, train_phenotypes)
    valid_phenotype_dict = init_phenotype_dict(mtl_tasks, valid_phenotypes)

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

    if subset_selection == 'outlier':
        # here we assume that class 0 is always the control class
        x_ctrl_train = X_train[y_train == 0]
        to_keep = int(0.1 * (X_train.shape[0] / len(train_phenotype_dict[0])))
        nsubset_ctrl = nsubset // np.sum(train_phenotype_dict[0] == 0)

        # generate a fixed number of subsets per class
        nsubset_biased = [0]
        for pheno in range(1, len(np.unique(train_phenotype_dict[0]))):
            nsubset_biased.append(nsubset // np.sum(train_phenotype_dict[0] == pheno))

        X_tr, y_tr = generate_biased_subsets_mtl(X_train, train_phenotype_dict.values(), id_train, x_ctrl_train,
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
            X_v, y_v = generate_biased_subsets_mtl(X_valid, valid_phenotype_dict.values(), id_valid, x_ctrl_valid,
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
        if per_sample:
            X_tr, y_tr = generate_subsets_mtl(X_train, list(train_phenotype_dict.values()), id_train,
                                              nsubset, ncell, per_sample)

            if (valid_samples is not None) or generate_valid_set:
                X_v, y_v = generate_subsets_mtl(X_valid, list(valid_phenotype_dict.values()), id_valid,
                                                nsubset, ncell, per_sample)

        # generate 'nsubset' multi-cell inputs per class
        else:
            nsubset_list = []
            for pheno in range(len(np.unique(train_phenotype_dict[0]))):
                nsubset_list.append(nsubset // np.sum(
                    train_phenotype_dict[0] == pheno))  # out of this we will pick the amount of subsets
            X_tr, y_tr = generate_subsets_mtl(X_train, list(train_phenotype_dict.values()), id_train,
                                              nsubset_list, ncell, per_sample)

            # check if listr elements usually differ from each other (if so we got a label imbalance here!). As well i play around with per_sample,
            if (valid_samples is not None) or generate_valid_set:
                nsubset_list = []
                for pheno in range(len(np.unique(valid_phenotype_dict[0]))):
                    nsubset_list.append(nsubset // np.sum(valid_phenotype_dict[0] == pheno))
                X_v, y_v = generate_subsets_mtl(X_valid, list(valid_phenotype_dict.values()), id_valid,
                                                nsubset_list, ncell, per_sample)
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
        y_tr[0] = tf.keras.utils.to_categorical(y_tr[0], n_classes)
        y_v[0] = tf.keras.utils.to_categorical(y_v[0], n_classes)

    # train some neural networks with different parameter configurations
    accuracies = np.zeros(nrun)
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
                            dropout, dropout_p, regression, n_classes, lr, mtl_tasks=mtl_tasks, loss_mode=loss_mode)
        print(model.summary())

        filepath = os.path.join(outdir, 'nnet_run_%d.hdf5' % irun)
        try:
            check = callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True,
                                              mode='auto')  # des saved the models
            earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='auto')
            loss_history = LossHistory(outdir=f'{outdir}/plots', irun=irun, loss_mode=loss_mode)
            mkdir_p('{outdir}/stats')
            csv_logger = CSVLogger(f'{outdir}/stats/training.csv', separator=',', append=True)
            tensorboard = TensorBoard(log_dir=f'{outdir}/stats/tensorboard/{model.name}_{time()}')

            y_trains = [y_tr[task] for task in range(mtl_tasks)]
            y_valids = [y_v[task] for task in range(mtl_tasks)]
            np.seterr('raise')

            if mtl_tasks == 1:
                history = model.fit(X_tr, y_trains[0],
                                    epochs=max_epochs, batch_size=bs,
                                    callbacks=[earlyStopping, check, tensorboard, csv_logger, loss_history],
                                    validation_data=(X_v, y_valids[0]), verbose=verbose)
            else:
                history = model.fit([X_tr, *y_trains], y_trains,
                                    epochs=max_epochs, batch_size=bs,
                                    callbacks=[earlyStopping, check, tensorboard, csv_logger, loss_history],
                                    validation_data=([X_v, *y_valids], y_valids), verbose=verbose)
            #tf.keras.utils.plot_model(model, to_file=f'{outdir}/plots/model_graph.png', show_shapes=True,
            #                         show_dtype=True, show_layer_activations=True, show_layer_names=True)
            plot_model_losses(history, f"{outdir}/stats/", irun)
            np.savetxt(f"{outdir}/stats/loss_history.txt", np.array(history.history['loss']), delimiter=",")

            # load the model from the epoch with highest validation accuracy
            model.load_weights(filepath)

            if mtl_tasks == 1:
                if not regression:
                    valid_metric = model.evaluate(X_v, *y_valids)
                    valid_metric_results = {i: metric for i, metric in enumerate(valid_metric)}
                    accuracies[irun] = valid_metric_results[1]  # accuracy
                else:
                    train_metric = model.evaluate(X_tr, *y_trains, batch_size=bs)
                    logger.info(f"Best train loss: {train_metric[1]:.2f}")
                    valid_metric = model.evaluate(X_v, *y_valids, batch_size=bs)
                    logger.info(f"Best validation loss: {valid_metric[1]:.2f}")
                    accuracies[irun] = - valid_metric[1] # MSE
            else:
                train_metric = model.evaluate([X_tr, *y_trains], y_trains, batch_size=bs)
                valid_metric = model.evaluate([X_v, *y_valids], y_valids, batch_size=bs)
                valid_metric_results = {i: metric for i, metric in enumerate(valid_metric)}
                train_metric_results = {i: metric for i, metric in enumerate(train_metric)}
                accuracies[irun] = - valid_metric_results[0]  # total loss

                for v_metric, t_metric, name in zip(valid_metric_results.values(), train_metric_results.values(),
                                                    model.metrics_names):
                    logger.info(f"Validation metric {name} achieved {v_metric:.2f}\n")
                    logger.info(f"Train metric {name} achieved {t_metric:.2f}\n")

            # extract the network parameters
            w_store[irun] = model.get_weights()

        except Exception as e:
            sys.stderr.write('An exception was raised during training the network.\n')
            sys.stderr.write(str(e) + '\n')

    # the top 3 performing networks
    model_sorted_idx = np.argsort(accuracies)[::-1][:3]
    best_3_nets = [w_store[i] for i in model_sorted_idx]
    best_net = best_3_nets[0]
    best_accuracy_idx = model_sorted_idx[0]

    # weights from the best-performing network
    # Not executeable with nfilters=[1]
    w_best_net = keras_param_vector(best_net)

    # post-process the learned filters
    # cluster weights from all networks that achieved losses above the specified thershold
    w_cons, cluster_res = cluster_profiles(w_store, nmark, accuracies, accur_thres,
                                           dendrogram_cutoff=dendrogram_cutoff)
    results = {
        'clustering_result': cluster_res,
        'selected_filters': w_cons,
        # filters that are representatives of the cluster (with members > 2) i = np.argmax(np.sum(pairwise_kernels(data, metric=metric), axis=1))
        'best_net': best_net,
        'best_3_nets': best_3_nets,
        'w_best_net': w_best_net,
        'accuracies': accuracies,
        'best_model_index': best_accuracy_idx,
        'config': config,
        'scaler': z_scaler,
        'n_classes': n_classes,
        'best_3_nets_ids': model_sorted_idx
    }

    if (valid_samples is not None) and (w_cons is not None):
        maxpool_percentage = config['maxpool_percentage'][best_accuracy_idx]
        for task in range(mtl_tasks):
            if isinstance(valid_phenotype_dict[task][0], int):
                # classification task
                filter_diff = get_filters_classification(w_cons, z_scaler,
                                                         valid_samples,
                                                         valid_phenotype_dict[task],
                                                         maxpool_percentage)

                if task == 0:  # this is mainly here for keeping the "old" output the same (STL way)
                    results['filter_diff'] = filter_diff
                else:
                    # if there ever will be an additional classification task
                    results[f'filter_diff_{task}'] = filter_diff
            else:
                # regression task
                tau = get_filters_regression(w_cons, z_scaler, valid_samples,
                                             valid_phenotype_dict[task],
                                             maxpool_percentage)

                if task == 0:  # this is mainly here for keeping the "old" output the same (STL way)
                    results['filter_tau'] = tau
                else:
                    results[f'filter_tau_{task}'] = tau
    return results


def init_phenotype_dict(mtl_tasks, phenotypes):
    dict = OrderedDict()
    if any(isinstance(item, np.ndarray) for item in phenotypes) or \
            any(isinstance(item, list) for item in phenotypes):
        ### MTL mode, check for list of lists
        if mtl_tasks != len(phenotypes):
            sys.stderr.write(f'"mtl_tasks" must be the length of the "phenotypes" list.\n')

        for task in range(mtl_tasks):
            dict[task] = np.asarray(phenotypes[task])
    else:
        # STL mode
        dict[0] = np.asarray(phenotypes)
    return dict


def build_model(ncell, nmark, nfilter, coeff_l1, coeff_l2,
                k, dropout, dropout_p, regression, n_classes, lr=0.01, mtl_tasks=1, loss_mode=None):
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

    # network prediction output
    losses, metrics, output_layers = [], [], []
    pheno_task_name = 'phenotype'
    if not regression:
        losses.append([tf.keras.losses.CategoricalCrossentropy()])  # for phenotype
        metrics.append(['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                        tf.keras.metrics.CategoricalCrossentropy()])
        # append regression frequency tasks
        if loss_mode == None or loss_mode.endswith('v2'):
            output = layers.Dense(units=n_classes,
                                  activation='softmax',
                                  kernel_initializer=initializers.RandomUniform(),
                                  kernel_regularizer=regularizers.l1_l2(l1=coeff_l1, l2=coeff_l2),
                                  name=f'{pheno_task_name}_cat_pred')(pooled)
        else:
            # Dense layer that wont add a loss to the model
            output = layers.Dense(units=n_classes, activation='softmax',
                                  name=f'{pheno_task_name}_cat_pred')(pooled)

        y_pheno = layers.Input(shape=(n_classes,), name=f'{pheno_task_name}_true')
    else:
        losses.append([tf.keras.losses.MeanSquaredError()])
        metrics.append(['mean_squared_error', tfa.metrics.r_square.RSquare(dtype=tf.float32, y_shape=(1,))])

        if loss_mode == None or loss_mode.endswith('v2'):
            output = layers.Dense(units=1,
                                  activation='linear',
                                  kernel_initializer=initializers.RandomUniform(),
                                  kernel_regularizer=regularizers.l1_l2(l1=coeff_l1, l2=coeff_l2),
                                  name=f'{pheno_task_name}_reg_pred')(pooled)
        else:
            output = layers.Dense(units=1, activation='linear',
                                  name=f'{pheno_task_name}_reg_pred')(pooled)
        y_pheno = layers.Input(shape=(1,), name=f'{pheno_task_name}_true')
    output_layers.append(output)

    ### this is the dummy for the freq regression for class this must be softmax as activation
    # all my atl tasks will be of regression type, start from 1 to not add the phenotype task twice
    y_task_inputs = dict()

    # the rest is filled with regression freq. tasks
    for task in range(1, mtl_tasks):
        taskname = f'input_task_{task}'
        losses.append([tf.keras.losses.MeanSquaredError()])
        metrics.append(['mean_squared_error', tfa.metrics.r_square.RSquare(dtype=tf.float32, y_shape=(1,))])
        if loss_mode == None or loss_mode.endswith('v2'):
            layer = layers.Dense(units=1,
                                 activation='linear',
                                 kernel_initializer=initializers.RandomUniform(),
                                 kernel_regularizer=regularizers.l1_l2(l1=coeff_l1, l2=coeff_l2),
                                 name=f'{taskname}_pred')(pooled)
        else:
            layer = layers.Dense(units=1, activation='linear',
                                 name=f'{taskname}_pred')(pooled)
        output_layers.append(layer)

        y_task_inputs[f'{taskname}_true'] = layers.Input(shape=(1,), name=f'{taskname}_true')

    # dynamically defining the inputs, the user needs to insert as many as tasks (obviously...)
    if mtl_tasks > 1:
        if loss_mode == 'uncertainty' or loss_mode == 'uncertainty_v2':
            ### implementation of Alex Kendall uncertainty weighting
            out = UncertaintyMultiLossLayer(loss_list=losses, nb_outputs=mtl_tasks)(
                [y_pheno, *y_task_inputs.values(), *output_layers])
            model = keras.Model(inputs=[data_input, y_pheno, *y_task_inputs.values()], outputs=out)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss=None,
                          metrics=metrics)
            assert len(model.layers[-1].trainable_weights) == mtl_tasks  # two log_vars, one for each output
            assert len(model.inputs) == mtl_tasks + 1  # X and the true labels
            if loss_mode == 'uncertainty':
                assert len(model.losses) == 2
            elif loss_mode == 'uncertainty_v2':
                assert len(model.losses) == 4

        if loss_mode == 'revised_uncertainty' or loss_mode == 'revised_uncertainty_v2':
            ### implementation of revised uncertainty weighting
            out = RevisedUncertaintyLossV2(loss_list=losses)([y_pheno, *y_task_inputs.values(), *output_layers])
            model = keras.Model(inputs=[data_input, y_pheno, *y_task_inputs.values()], outputs=out)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss=None,
                          metrics=metrics)

            assert len(model.layers[-1].trainable_weights) == mtl_tasks  # two log_vars, one for each output
            assert len(model.inputs) == mtl_tasks + 1  # X and the true labels
            if loss_mode == 'revised_uncertainty':
                assert len(model.losses) == 2
            elif loss_mode == 'revised_uncertainty_v2':
                assert len(model.losses) == 4
        if loss_mode == None:
            # this is mostly there to not need to change the amount of inpt parameters
            model = keras.Model(inputs=[data_input, y_pheno, *y_task_inputs.values()], outputs=output_layers)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss=losses,
                          metrics=metrics)
    else:
        model = keras.Model(inputs=data_input, outputs=output_layers)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=losses,
                      metrics=metrics)
    # metrics are a list of lists, first LIST is for first output and so on...
    return model


def pool_top_k(x, k):
    return tf.reduce_mean(tf.sort(x, axis=1, direction='DESCENDING')[:, :k, :], axis=1)
