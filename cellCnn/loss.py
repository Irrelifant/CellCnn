import numpy as np
from keras.layers import Layer
from keras import backend as K
import tensorflow as tf


# inheriting from Layer would allow automated gradient backpropagation that is not available for Callback
class RevisedUncertaintyLoss(Layer):
    def __init__(self, loss_list, *args, **kwargs):
        super(RevisedUncertaintyLoss, self).__init__(**kwargs)
        self.loss_list = loss_list

    def build(self, input_shape=None):
        print('Build called')
        self.sigmas_sq = []
        for i in range(len(self.loss_list)):
            # random_uniform ?? initial var
            self.sigmas_sq += [self.add_weight(name=f'sigmas_sq_{i}', shape=(1,),
                                               initializer=tf.initializers.RandomUniform(minval=0.2, maxval=1),
                                               trainable=True).numpy()]

        # self.sigmas_sq = np.array(sigma_list)
        super(RevisedUncertaintyLoss, self).build(input_shape)

    #todo find out if it makes any difference, same for reduce min, und bei tf.add statt +
    def get_mtl_loss(self, ys_true, ys_pred):
        print('in mtl: split inputs')
        assert len(ys_true) == len(self.loss_list)
        assert len(ys_pred) == len(self.loss_list)
        loss = 0.
        # evtl die richtigen loss funktions nehmen...
        for i in range(0, len(self.loss_list)):
            sigma_sq = tf.pow(self.sigmas_sq[i], 2)
            factor = tf.math.divide_no_nan(1.0, tf.multiply(2.0, sigma_sq))
            listed_loss = self.loss_list[i](ys_true[i], ys_pred[i])
            loss = tf.add(loss, tf.add(tf.multiply(factor, listed_loss), tf.math.log(tf.add(1, sigma_sq))))
        print('mtl loss end')
        print(loss)
        return loss # tf.reduce_min

    def get_config(self):
        config = super().get_config()
        config.update({
            "loss_list": self.loss_list,
            "sigmas_sq": self.sigmas_sq,
        })
        return config

    def call(self, inputs, *args, **kwargs):
        print('CALLING...')
        print(inputs)
        print('Sigmas')
        print(self.sigmas_sq)

        ys_true = inputs[:len(self.loss_list)]
        ys_pred = inputs[len(self.loss_list):]
        loss = self.get_mtl_loss(ys_true, ys_pred)
        print('loss')
        print(loss)
        self.add_loss(loss, inputs=inputs)
        return ys_pred  # this doesn´t really matter -> apparently it does since that is what my out will be...
