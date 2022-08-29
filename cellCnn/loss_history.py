import os
import keras.callbacks
from matplotlib import pyplot as plt

class LossHistory(keras.callbacks.Callback):
    def __init__(self, outdir='/default_outdir', irun=0, loss_mode=None):
        self.irun = irun
        self.plotdir = outdir
        self.loss_mode = loss_mode
        os.makedirs(self.plotdir, exist_ok=True)

    def on_train_begin(self, logs=None):
        self.sigmas = []
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        if hasattr(self.model.layers[-1], 'get_sigmas'):
            sigmas = [s.numpy() for s in self.model.layers[-1].get_sigmas()]
            self.sigmas.append(sigmas)
        self.losses.append(logs)

    def on_train_end(self, logs=None):
        plt.clf()  # clear

        if len(self.sigmas) > 0:
            for i, _ in enumerate(self.sigmas[0]):
                sigmas = [s[i] for s in self.sigmas]
                plt.plot(range(len(sigmas)), sigmas,
                         label=f'sigma {i}')
            if self.loss_mode == 'uncertainty' or self.loss_mode == 'uncertainty_v2':
                plt.title(f'Log Sigmas along epochs')
                plt.ylabel('Log Sigma')
            else:
                plt.title(f'Sigma Squares along epochs')
                plt.ylabel('Sigma Square')

            plt.xlabel(f'Epochs')
            plt.legend()
            plt.savefig(f'{self.plotdir}/sigma_plot_{self.irun}.png')
            plt.close()