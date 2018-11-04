import tensorflow as tf

import baleian.tf.nn.optimizable_neural_net as nn


class LinearRegression(nn.OptimizableNetwork):

    def __init__(self, input_size, name=None):
        super(LinearRegression, self).__init__(input_size, name)
        self._Y = None
        self._cost = None
        self._train = None
        self.set_optimizer(tf.train.GradientDescentOptimizer)

    def _on_model_changed(self):
        self._Y = tf.placeholder(tf.float32, [None, self._output_size])
        self._cost = tf.reduce_mean(tf.square(self._H - self._Y))
        self._train = self._optimizer(self._learning_rate).minimize(self._cost)

    def add_layer(self, output_size, **kwargs):
        return super(LinearRegression, self) \
            .add_layer(output_size, layer=tf.layers.dense, **kwargs)

    def session(self, sess: tf.Session):
        return ModelSession(self, sess)


# noinspection PyProtectedMember
class ModelSession(nn.ModelSession):

    def __init__(self, net: LinearRegression, sess: tf.Session):
        super(ModelSession, self).__init__(net, sess)

    def train(self, x_data, y_data):
        """Functional interface for training this model

        :param x_data:
        :param y_data:
        :return:
        """
        cost = self.sess.run([self.net._cost, self.net._train],
                             feed_dict={
                                 self.net._X: x_data,
                                 self.net._Y: y_data
                             })
        return cost

    def test(self, x_data, y_data):
        """Functional interface for obtaining the predicted value

        :param x_data:
        :param y_data:
        :return:
        """
        return self.sess.run([self.net._pred, self.net._cost],
                             feed_dict={
                                 self.net._X: x_data,
                                 self.net._Y: y_data
                             })
