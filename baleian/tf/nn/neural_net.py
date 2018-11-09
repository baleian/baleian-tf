from abc import *
import tensorflow as tf


class AbstractNeuralNetwork(metaclass=ABCMeta):

    def __init__(self, input_size):
        self.input_size = input_size
        self.input_layers = []
        self.hidden_layers = []
        self.output_layers = []

    def add_input_layer(self, output_size,
                        layer: type(tf.layers.Layer)=tf.layers.Dense,
                        **kwargs):
        self.input_layers.append((output_size, layer, kwargs))

    def add_hidden_layer(self, output_size,
                         layer: type(tf.layers.Layer)=tf.layers.Dense,
                         **kwargs):
        self.hidden_layers.append((output_size, layer, kwargs))

    def add_output_layer(self, output_size,
                         layer: type(tf.layers.Layer)=tf.layers.Dense,
                         **kwargs):
        self.output_layers.append((output_size, layer, kwargs))

    @abstractmethod
    def session(self, *args, **kwargs):
        pass


class AbstractModelSession(metaclass=ABCMeta):

    def __init__(self, sess: tf.Session, name=None, **kwargs):
        self.sess = sess
        self.name = name
        self.input_size = None
        self.output_size = None
        self.X = None
        self.H = None
        self._layers = []
        self.layers = []
        self.pred = None

    def init_layer(self, net: AbstractNeuralNetwork):
        with tf.variable_scope(self.name):
            self.input_size = net.input_size
            self.output_size = net.input_size
            self.X = tf.placeholder(tf.float32, [None, self.input_size])
            self.H = self.X
            for output_size, layer, kwargs in (
                    net.input_layers + net.hidden_layers + net.output_layers):
                _layer = layer(output_size, **kwargs)
                self.output_size = output_size
                self.H = _layer.apply(self.H)
                self._layers.append(_layer)
                self.layers.append(self.H)
            self.sess.run(tf.global_variables_initializer())

    @abstractmethod
    def init_model(self):
        """Overridable function for model generation

        This function will be called when model's network is organized.
        Used in subclasses to set operations based on layer changes.

        :return:
        """
        pass

    def prediction(self, x_data):
        return self.sess.run(self.pred, feed_dict={self.X: x_data})

    def get_weight(self, layer_name):
        with tf.variable_scope(self.name):
            with tf.variable_scope(layer_name, reuse=True):
                return self.sess.run(tf.get_variable("kernel"))


class ModelSession(AbstractModelSession):

    def init_model(self):
        self.pred = self.H


class NeuralNetwork(AbstractNeuralNetwork):

    def session(self, sess: tf.Session, name=None,
                model: type(AbstractModelSession)=ModelSession,
                **kwargs):
        instance = object.__new__(model)
        instance.__init__(sess, name, **kwargs)
        instance.init_layer(self)
        instance.init_model()
        return instance
