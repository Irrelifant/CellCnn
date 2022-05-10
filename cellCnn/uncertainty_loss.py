####
## Uncertainty loss layer from https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example.ipynb
## refered from https://towardsdatascience.com/deep-multi-task-learning-3-lessons-learned-7d0193d71fd6
##
from keras import backend as K
from keras.initializers.initializers_v2 import Constant
from keras.layers import Layer
import tensorflow as tf

# Custom loss layer
class UncertaintyMultiLossLayer(Layer):
    def __init__(self, loss_list=None, nb_outputs=2, **kwargs):
        self.loss_list = loss_list
        self.nb_outputs = nb_outputs
        super(UncertaintyMultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        self.log_vars = [] # container of log sigma squares
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=tf.keras.initializers.Constant(0.),
                                              trainable=True)]
        super(UncertaintyMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_variance, task_loss in zip(ys_true, ys_pred, self.log_vars, self.loss_list):
            precision = K.exp(-log_variance[0]) # resolved to the positive domain giving valid values for variance.
            listed_loss = task_loss[0](y_true, y_pred)
            loss += tf.add(precision * listed_loss + log_variance[0], -1)
        return K.mean(loss)

    def get_config(self):
        config = super().get_config()
        # https://github.com/tensorflow/tensorflow/issues/28799 describes that you can only serialize numpys when saving the model (later we will save...)
        config.update({
            "nb_outputs": self.nb_outputs,
            "loss_list": self.loss_list
        })
        return config

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        self.add_metric(loss, name='uncertainty_loss')
        return ys_pred