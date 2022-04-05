import os
import keras.callbacks
from matplotlib import pyplot as plt

#todo extend this further
class LossHistory(keras.callbacks.Callback):
    def __init__(self, outdir='/default_outdir', irun=0):
        self.irun = irun
        self.plotdir = outdir
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
            plt.title(f'Sigmas along epochs')
            plt.xlabel(f'Epochs')
            plt.ylabel('Sigma')
            plt.legend()
            plt.savefig(f'{self.plotdir}/sigma_plot_{self.irun}.png')
