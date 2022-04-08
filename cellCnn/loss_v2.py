from keras.layers import Layer
import tensorflow as tf

#todo add description to loss layer

# inheriting from Layer would allow automated gradient backpropagation that is not available for Callback
# similar to kendall´´ implementation
def get_listed_loss_by_shape(y_true, y_pred):
    if y_true.shape[1] == 2:
        # classification
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        return tf.reduce_mean(cross_entropy, name='loss')
    else:
        # regression
        return tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(y_pred, y_true)), axis=-1))
        # axis = -1 means to return index of last axis


class RevisedUncertaintyLossV2(Layer):
    def __init__(self, loss_list, sigmas, *args, **kwargs):
        super(RevisedUncertaintyLossV2, self).__init__()
        self.loss_list = loss_list
        self.sigmas = sigmas

    def get_sigmas(self):
        return self.sigmas

    def get_mtl_loss(self, ys_true, ys_pred):
        print('in mtl: split inputs')
        assert len(ys_true) == len(self.loss_list)
        assert len(ys_pred) == len(self.loss_list)
        loss = 0.
        for i in range(0, len(self.loss_list)):
            sigma_sq = tf.pow(self.sigmas[i], 2)
            factor = tf.math.divide_no_nan(1.0, tf.multiply(2.0, sigma_sq))

            listed_loss = get_listed_loss_by_shape(ys_true[i], ys_pred[i])
            # listed_loss = self.loss_list[i](ys_true[i], ys_pred[i])

            loss = tf.add(loss, tf.add(tf.multiply(factor, listed_loss), tf.math.log(tf.add(1., sigma_sq))))
        return loss

    def get_config(self):
        config = super().get_config()
        # https://github.com/tensorflow/tensorflow/issues/28799 describes that you can only serialize numpys when saving the model (later we will save...)
        sigmas_to_save = [s.numpy() for s in self.sigmas]
        config.update({
            "loss_list": self.loss_list,
            "sigmas": sigmas_to_save
        })
        return config

    def call(self, inputs, *args, **kwargs):
        print('CALLING...')
        ys_true = inputs[:len(self.loss_list)]
        ys_pred = inputs[len(self.loss_list):]
        loss = self.get_mtl_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        self.add_metric(loss, name='revised_uncertainty_loss')
        return ys_pred
