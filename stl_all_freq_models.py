import gc
import glob
import importlib
import pickle
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
### (from: https://github.com/eiriniar/CellCnn/blob/0413a9f49fe0831c8fe3280957fb341f9e028d2d/cellCnn/examples/NK_cell_ungated.ipynb ) AND https://github.com/eiriniar/CellCnn/blob/0413a9f49fe0831c8fe3280957fb341f9e028d2d/cellCnn/examples/PBMC.ipynb
import pandas as pd
import numpy as np
import seaborn as sns
from cellCnn.ms.utils.helpers import calc_frequencies
from cellCnn.ms.utils.helpers import get_fitted_model, split_test_train_valid
from cellCnn.plotting import plot_results, plot_tsne_selection_grid
from cellCnn.plotting import plot_selected_cells_for_filter_tsne, plot_filters, discriminative_filters
from cellCnn.utils import save_results, get_selected_cells, mkdir_p
from sklearn.manifold import TSNE
import os
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import tensorflow as tf

def main():
    ##### state vars
    cytokines = ['CCR2', 'CCR4', 'CCR6', 'CCR7', 'CXCR4', 'CXCR5', 'CD103', 'CD14', 'CD20', 'CD25', 'CD27', 'CD28', 'CD3',
                 'CD4', 'CD45RA', 'CD45RO', 'CD56', 'CD57', 'CD69', 'CD8', 'TCRgd', 'PD.1', 'GM.CSF', 'IFN.g', 'IL.10',
                 'IL.13', 'IL.17A', 'IL.2', 'IL.21', 'IL.22', 'IL.3', 'IL.4', 'IL.6', 'IL.9', 'TNF.a']
    infile = 'cohort_denoised_clustered_diagnosis_patients.csv'
    indir = 'cellCnn/ms/data/input'
    outdir = 'mtl_models_t2'
    rand_seed = 123
    train_perc = 0.7
    test_perc = 0.3
    ## information from ms_data project
    cluster_to_celltype_dict = {0: 'b', 1: 'cd4', 3: 'nkt', 8: 'cd8', 10: 'nk', 11: 'my', 16: 'dg'}

    np.random.seed(rand_seed)
    mkdir_p(outdir)
    df = pd.read_csv(os.path.join(indir, infile), index_col=0)
    df = df.drop_duplicates()  ### reduces overfitting at cost of fewer data
    rrms_df = df[df['diagnosis'] == 'RRMS']
    rrms_patients2df = {id: patient_df.drop(columns=['diagnosis', 'gate_source']) for id, patient_df in
                        rrms_df.groupby('gate_source')}
    nindc_df = df[df['diagnosis'] == 'NINDC']
    nindc_patients2df = {id: patient_df.drop(columns=['diagnosis', 'gate_source']) for id, patient_df in
                         nindc_df.groupby('gate_source')}
    #### here we could see freq differences across the 2 groups
    print('Frequencies: ')
    rrms_patients_freq = {id: calc_frequencies(patient_df, cluster_to_celltype_dict, return_list=True) for id, patient_df in
                          rrms_patients2df.items()}
    nindc_patients_freq = {id: calc_frequencies(patient_df, cluster_to_celltype_dict, return_list=True) for id, patient_df
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

    X = [selection[0].to_numpy() for selection in all_chunks]
    freqs = [selection[1] for selection in all_chunks]
    Y = [selection[2] for selection in all_chunks]
    print('DONE: batches created')

    # for regression task stratified is wrong since there are no classes
    X_test, X_train, X_valid, freq_test, freq_train, freq_valid, y_test, y_train, y_valid = split_test_train_valid(
        X, freqs, Y,
        train_perc=train_perc,
        test_perc=test_perc,
        valid_perc=0.6)

    for i, freq_idx in enumerate(cluster_to_celltype_dict.keys()):
        if i ==0:
            continue
        gc.collect()

        outdir = f'stl_models/stl_{cluster_to_celltype_dict[freq_idx]}'
        print('Getting the proper freq- data from freq arrays')
        cell_type_freq_test = [series[i] for series in freq_test]
        cell_type_freq_train = [series[i] for series in freq_train]
        cell_type_freq_valid = [series[i] for series in freq_valid]
        print(f'Building STL model for {freq_idx} ({cluster_to_celltype_dict[freq_idx]})... ')

        os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
        os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async' 
        #os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus, 'GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
                print(e)    
        #os.environ["CUDA_VISIBLE_DEVICES"]="0"

        # nsubset parameter to 1000 as in Galli
        # pay attention to put in phenotype first and then freq...
        model = get_fitted_model(X_train, X_valid, cell_type_freq_train, cell_type_freq_valid,
                nsubset=1000, nrun=15, nfilters=[4],
                                 regression=True,
                                 per_sample=True, quant_normed=True, scale=False,
                                 coeff_l1=0, max_epochs=20, learning_rate=None,
                                 ncell=2000, subset_selection='random',
                                 outdir=outdir, verbose=True)
        results = model.results
        pickle.dump(results, open(os.path.join(outdir, 'results.pkl'), 'wb'))
#        save_results(results, outdir, cytokines)

        metric_results_df = pd.DataFrame()
        pred_results = model.predict(X_test)
        #freq_test = [freq[i] for freq in cell_type_freq_test]
        mse_freq = mean_squared_error(cell_type_freq_test, pred_results)
        metric_results_df['freq_mse'] = [mse_freq]

        r2_freq = r2_score(cell_type_freq_test, pred_results)
        metric_results_df['freq_r2'] = [r2_freq]
        metric_results_df.to_csv(outdir + '/test_stats.csv')
        print(f'Done test stats to {outdir}')

    print('DONE building models')

    cluster = df.loc[:, 'cluster'].astype(int).reset_index(drop=True)
    x_name = 'cohort'
    x = df.iloc[:, :len(cytokines)].reset_index(drop=True)

    # x_for_tsne = x.iloc[np.random.choice(x.shape[0], 1000), :]
    x_tsne = TSNE(n_components=2).fit_transform(x)
    x_tsne_df = pd.DataFrame(x_tsne)


    for i, freq_idx in enumerate(cluster_to_celltype_dict.keys()):
        outdir = f'stl_models/stl_{cluster_to_celltype_dict[freq_idx]}'
        plotdir = os.path.join(outdir, 'plots')
        results = pickle.load(open(os.path.join(outdir, 'results.pkl'), 'rb'))
        cell_type_freq_test = [series[i] for series in freq_test]
        cell_type_freq_train = [series[i] for series in freq_train]
        cell_type_freq_valid = [series[i] for series in freq_valid]
        if results['selected_filters'] is not None:
            _v = discriminative_filters(results, os.path.join(plotdir, 'filter_plots_discriminative'),
                                        filter_diff_thres=0.2, show_filters=True)

            # for regression, make a biaxial plot (phenotype vs. frequency)
            filter_info = plot_results(results, X_train, cell_type_freq_train,
                                       cytokines, os.path.join(plotdir, 'training_plots'),
                                       filter_diff_thres=0.2,
                                       filter_response_thres=0,
                                       stat_test='mannwhitneyu',
                                       tsne_ncell=1000,
                                       regression=True,
                                       show_filters=True)
            _v = plot_results(results, X_valid, cell_type_freq_valid,
                              cytokines, os.path.join(plotdir, 'validation_plots'),
                              filter_diff_thres=0.2,
                              filter_response_thres=0,
                              stat_test='mannwhitneyu',
                              tsne_ncell=1000, regression=False,
                              show_filters=True)
            filters_dir = f'{outdir}/selected_cells/filters'
            abundancy_dir = f'{filters_dir}/abundancies'
            mkdir_p(filters_dir)
            mkdir_p(abundancy_dir)

            # outdir = f'stl_models/stl_{cluster_to_celltype_dict[freq_idx]}'
            #plotdir = os.path.join(outdir, 'plots')
            # results = pickle.load(open(os.path.join(outdir, 'results.pkl'), 'rb'))

            print('Available filters:')
            print(filter_info)
            print('Plotting the bar abundancy related stuff..')
            flags = np.zeros((x.shape[0], 2 * len(filter_info)))
            columns = []
            for i, (filter_idx, thres) in enumerate(filter_info):
                # [:, 2 * i:2 * (i + 1)] basically extends the flas array by its filters values (2 cols per filter)
                # thres is bissle fürn arsch ... weil hier bekomm ich mehr zellen raus wenn ich die runtersetze...
                cells = get_selected_cells(results['selected_filters'][filter_idx], np.asarray(x), results['scaler'], thres, True)
                flags[:, 2 * i:2 * (i + 1)] = cells
                columns += ['filter_%d_continuous' % filter_idx, 'filter_%d_binary' % filter_idx]

                # per filter calculate relative abundancies for filters, save them
                # is always of size 3
                flags_df = pd.DataFrame(flags[:, 2 * i:2 * (i + 1)])
                flags_df['cluster'] = cluster
                flags_df.to_csv(os.path.join(filters_dir, f'filter_{filter_idx}_selected_cells_w_clusters.csv'), index=False)

                # SAVE relative abundance values
                selected_cells_filter = flags_df[flags_df.loc[:, 0] != 0]
                # absolute values
                selected_cells_filter_grpd_tot = selected_cells_filter.groupby('cluster').count()[0]
                #values relative to selected cell size
                selected_cells_filter_grpd = selected_cells_filter_grpd_tot / selected_cells_filter.shape[0]
                selected_cells_filter_grpd.to_csv(os.path.join(abundancy_dir, f'filter_{filter_idx}_cell_type_abundancies.csv'),
                                                  index=True)

                #############################################
                ### plot t-SNE of the cells from this filter:
                ### most code taken from plot_results() as there is already a similar solution
                # there x relates to the input data (e.g. x_train, y_train ... i guess i can just take the whole cohort)
                plot_selected_cells_for_filter_tsne(x_tsne_df, selected_cells_filter, filter_idx, cluster,
                                                    abundancy_dir=abundancy_dir, )

            df = pd.DataFrame(flags, columns=columns)
            df.to_csv(os.path.join(outdir, f'selected_cells/{x_name}_selected_cells.csv'), index=False)
            print(f'done saving selected cells to {outdir}')

            #### BARPLOT comparing the filters
            ### pitch for plotting all the abundancies of all selected filters and plot them aside to compare within a model
            files = glob.glob(abundancy_dir + '/*.csv')
            abundancy_dfs = [pd.read_csv(filename, index_col=0, header=0) for filename in files]
            abundancy_df = pd.concat(abundancy_dfs, axis=1, ignore_index=True)
            abundancy_df.columns = [filename.split('/')[-1] for filename in files]
            abundancy_df = abundancy_df.reset_index()
            abundancy_dfm = abundancy_df.melt('cluster', var_name='cols', value_name='vals')

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            sns.barplot(x='cluster', y='vals', hue='cols', data=abundancy_dfm)
            ax.set_xticklabels(list(cluster_to_celltype_dict.values()))
            plt.title('MTL')
            plt.savefig(f'{abundancy_dir}/comparison_barplot.png')
            plt.close()
    print('done')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupt!\n")
        sys.exit(-1)
