from keras.layers import Layer
import tensorflow as tf


# inheriting from Layer would allow automated gradient backpropagation that is not available for Callback
def get_listed_loss_by_shape(y_true, y_pred):
    if y_true.shape[1] == 2:
        # classification
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        return tf.reduce_mean(cross_entropy, name='loss')
    else:
        return tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(y_pred, y_true)), axis=-1))


class RevisedUncertaintyLoss(Layer):
    def __init__(self, loss_list, *args, **kwargs):
        super(RevisedUncertaintyLoss, self).__init__()
        self.loss_list = loss_list

    def build(self, input_shape=None):
        print('Build called')
        self.sigmas = []
        sigma_init = tf.random_uniform_initializer(minval=0.2, maxval=1.)
        for i in range(len(self.loss_list)):
            sigma = tf.Variable(name=f'sigmas_sq_{i}', dtype=tf.float32,
                                initial_value=sigma_init(shape=(),
                                                         dtype='float32'),
                                trainable=True)
            self.sigmas.append(sigma)

    def get_sigmas(self):
        return self.sigmas

    def get_mtl_loss(self, ys_true, ys_pred):
        print('in mtl: split inputs')
        assert len(ys_true) == len(self.loss_list)
        assert len(ys_pred) == len(self.loss_list)
        loss = 0.
        # evtl die richtigen loss funktions nehmen...
        for i in range(0, len(self.loss_list)):
            sigma_sq = tf.pow(self.sigmas[i], 2)
            print(f'sigma {i}')
            print(self.sigmas[i])
            factor = tf.math.divide_no_nan(1.0, tf.multiply(2.0, sigma_sq))

            listed_loss = get_listed_loss_by_shape(ys_true[i], ys_pred[i])
            # listed_loss = self.loss_list[i](ys_true[i], ys_pred[i])
            print(f'listed_loss task {i}')
            print(listed_loss)

            loss = tf.add(loss, tf.add(tf.multiply(factor, listed_loss), tf.math.log(tf.add(1., sigma_sq))))
        return loss

    #   without this reduce min i would get a shape error. gelling me to pass shapes of fitting ranks in
    # todo investigate

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
        print('loss')
        print(loss)
        self.add_loss(loss, inputs=inputs)
        return ys_pred  # this doesn´t really matter -> apparently it does since that is what my out will be...
