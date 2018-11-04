import tensorflow as tf


class NeuralNetwork:

    def __init__(self, input_size, name=None):
        self._name = name if name is not None else NeuralNetwork.__name__
        self._input_size = input_size
        self._output_size = input_size
        self._X = tf.placeholder(tf.float32, [None, input_size])
        self._H = self._X  # 0 depth layer
        self._pred = self._H

    def add_layer(self, output_size,
                  layer: tf.layers.Layer,
                  activation=None,
                  name=None):
        with tf.variable_scope(self._name):
            self._H = layer(self._H, output_size,
                            activation=activation,
                            name=name)
        self._output_size = output_size
        self._pred = self._H
        return self

    def _on_model_changed(self):
        """Overridable function for layer reorganization

        This function will be called when model's network is reorganized.
        Used in subclasses to reset operations based on layer changes.

        :return:
        """
        self._pred = self._H

    def session(self, sess: tf.Session):
        return ModelSession(self, sess)


class ModelSession:

    def __init__(self, net: NeuralNetwork, sess: tf.Session):
        self.net = net
        self.sess = sess
        self.net._on_model_changed()

    def add_layer(self, output_size,
                  layer: tf.layers.Layer,
                  activation=None,
                  name=None):
        arguments = locals()
        del arguments["self"]
        self.net.add_layer(**arguments)
        self.net._on_model_changed()
        self.sess.run(tf.global_variables_initializer())

    def prediction(self, features):
        """Functional interface for obtaining the predicted value

        :param features:
        :return:
        """
        return self.sess.run(self.net._pred, feed_dict={self.net._X: features})

    def get_weight(self, layer_name):
        with tf.variable_scope(self.net._name):
            with tf.variable_scope(layer_name, reuse=True):
                return self.sess.run(tf.get_variable("kernel"))
