import tensorflow as tf

import baleian.tf.nn.neural_net as nn


class OptimizableNetwork(nn.NeuralNetwork):

    def __init__(self, input_size, name=None):
        super(OptimizableNetwork, self).__init__(input_size, name)
        self._optimizer = None
        self._learning_rate = None

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer
        return self

    def set_learning_rate(self, learning_rate):
        self._learning_rate = learning_rate
        return self

    def session(self, sess: tf.Session):
        return ModelSession(self, sess)


# noinspection PyProtectedMember
class ModelSession(nn.ModelSession):

    def __init__(self, net: OptimizableNetwork, sess: tf.Session):
        super(ModelSession, self).__init__(net, sess)

    def set_optimizer(self, optimizer):
        self.net.set_optimizer(optimizer)
        self.net._on_model_changed()

    def set_learning_rate(self, learning_rate):
        self.net.set_learning_rate(learning_rate)
        self.net._on_model_changed()
