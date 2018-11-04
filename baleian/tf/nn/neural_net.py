from abc import *
import tensorflow as tf


class NeuralNetwork(metaclass=ABCMeta):

    def __init__(self, input_size, name=None):
        self._name = name
        self._X = tf.placeholder(tf.float32, [None, input_size])
        self._H = self._X  # 0 depth layer
        self._Y = tf.placeholder(tf.float32, [None, input_size])
        self._pred = self._H

    @abstractmethod
    def _on_layer_generated(self):
        pass

    def add_dense_layer(self, output_size, activation=None, name=None):
        with tf.variable_scope(self._name, NeuralNetwork.__name__):
            self._H = tf.layers.dense(self._H, output_size,
                                     activation=activation,
                                     name=name)
        self._Y = tf.placeholder(tf.float32, [None, output_size])
        self._pred = self._H
        self._on_layer_generated()

    def prediction(self, sess: tf.Session, x_data):
        return sess.run(self._pred, feed_dict={self._X: x_data})

    def get_weight(self, sess: tf.Session, layer_name):
        with tf.variable_scope(self._name, NeuralNetwork.__name__):
            with tf.variable_scope(layer_name, reuse=True):
                return sess.run(tf.get_variable("kernel"))
