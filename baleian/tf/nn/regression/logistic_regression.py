import baleian.tf.nn.supervised_learning_net as nn

import tensorflow as tf


class LogisticRegression(nn.OptimizableNetwork):

    def __init__(self, input_size):
        super(LogisticRegression, self).__init__(input_size)
        self.add_output_layer(1, activation=tf.sigmoid, name="output")
        self.set_optimizer(tf.train.GradientDescentOptimizer)
        self.set_learning_rate(1e-1)

    def session(self, *args, **kwargs):
        return super(LogisticRegression, self) \
            .session(model=ModelSession, *args, **kwargs)


class ModelSession(nn.AbstractModelSession):

    def __init__(self, *args, **kwargs):
        super(ModelSession, self).__init__(*args, **kwargs)
        self._Y = None
        self.accuracy = None
        self.validity = None

    def init_model(self):
        self.Y = tf.placeholder(tf.int32, [None, ])
        self._Y = tf.reshape(tf.cast(self.Y, dtype=tf.float32), [-1, 1])
        self.pred = tf.reshape(tf.cast(self.H > 0.5, dtype=tf.int32), [-1])
        self.cost = -tf.reduce_mean(
            self._Y * tf.log(self.H) + (1 - self._Y) * tf.log(1 - self.H)
        )
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.pred, self.Y), dtype=tf.float32)
        )
        self.validity = tf.reshape(tf.abs(self.H - 0.5) * 2, [-1])

    def get_accuracy(self, x_data, y_data):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_data, self.Y: y_data})

    def get_validity(self, x_data, y_data):
        return self.sess.run(self.validity,
                             feed_dict={self.X: x_data, self.Y: y_data})
