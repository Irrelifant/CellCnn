import tensorflow as tf
from keras import backend as K
from keras.layers import Layer


""" 
Author: Elias Schreiner

This is a custom layer that should implement the Revised Uncertainty Loss (https://github.com/Mikoto10032/AutomaticWeightedLoss) 
which is an extension to the uncertainty weighting by Kendall (https://github.com/ranandalon/mtl)
"""

class RevisedUncertaintyLossV2(Layer):
    def __init__(self, loss_list, **kwargs):
        self.loss_list = loss_list
        self.sigmas_sq = []
        super(RevisedUncertaintyLossV2, self).__init__(**kwargs)

    def build(self, input_shape=None):
        self.sigmas_sq = []
        const_init = tf.keras.initializers.Constant(1 / len(self.loss_list))
        for i in range(len(self.loss_list)):
            self.sigmas_sq += [self.add_weight(name=f'sigma_{i}', shape=(1,),
                                               initializer=const_init,
                                               trainable=True)]
        super(RevisedUncertaintyLossV2, self).build(input_shape)

    def get_sigmas(self):
        return self.sigmas_sq

    def get_mtl_loss(self, ys_true, ys_pred):
        assert len(ys_true) == len(self.sigmas_sq)
        assert len(ys_pred) == len(self.sigmas_sq)
        loss = 0.
        for y_true, y_pred, sigma_sq, task_loss in zip(ys_true, ys_pred, self.sigmas_sq, self.loss_list):
            factor = tf.math.divide_no_nan(1.0, tf.multiply(2.0, sigma_sq))
            listed_loss = task_loss[0](y_true, y_pred)
            tf.print(sigma_sq, listed_loss, factor, tf.math.log(1 + sigma_sq), sep='_;_')
            loss += K.sum(factor * listed_loss + tf.math.log(1 + sigma_sq))
        return K.mean(loss)

    def get_config(self):
        config = super().get_config()
        # https://github.com/tensorflow/tensorflow/issues/28799 describes that you can only serialize numpys when saving the model (later we will save...)
        sigmas_to_save = [s.numpy() for s in self.sigmas_sq]
        config.update({
            "sigmas": sigmas_to_save,
            "loss_list": self.loss_list
        })
        return config

    def call(self, inputs, *args, **kwargs):
        ys_true = inputs[:len(self.sigmas_sq)]
        ys_pred = inputs[len(self.sigmas_sq):]
        loss = self.get_mtl_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        self.add_metric(loss, name='revised_uncertainty_loss')
        #[print(sigma_sq) for sigma_sq in self.sigmas_sq]
        return ys_pred
