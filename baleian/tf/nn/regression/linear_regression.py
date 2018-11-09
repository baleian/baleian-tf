import baleian.tf.nn.supervised_learning_net as nn

import tensorflow as tf


class LinearRegression(nn.OptimizableNetwork):

    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__(input_size)
        self.add_output_layer(output_size, name="output")
        self.set_optimizer(tf.train.GradientDescentOptimizer)
        self.set_learning_rate(1e-6)

    def session(self, *args, **kwargs):
        return super(LinearRegression, self) \
            .session(model=ModelSession, *args, **kwargs)


class ModelSession(nn.AbstractModelSession):

    def init_model(self):
        self.pred = self.H
        self.cost = tf.reduce_mean(tf.square(self.pred - self.Y))
