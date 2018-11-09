from baleian.tf.nn.optimizable_net import *

from abc import *
import tensorflow as tf


class AbstractModelSession(AbstractModelSession, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super(AbstractModelSession, self).__init__(*args, **kwargs)
        self.Y = None

    def init_layer(self, net: OptimizableNetwork):
        super(AbstractModelSession, self).init_layer(net)
        self.Y = tf.placeholder(tf.float32, [None, self.output_size])

    def training(self, x_data, y_data):
        self.sess.run(self.optimize,
                      feed_dict={self.X: x_data, self.Y: y_data})

    def get_cost(self, x_data, y_data):
        return self.sess.run(self.cost,
                             feed_dict={self.X: x_data, self.Y: y_data})
