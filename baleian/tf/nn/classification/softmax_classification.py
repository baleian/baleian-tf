import baleian.tf.nn.supervised_learning_net as nn

import tensorflow as tf


class SoftmaxClassification(nn.OptimizableNetwork):

    def __init__(self, input_size, label_size):
        super(SoftmaxClassification, self).__init__(input_size)
        self.label_size = label_size
        self.with_one_hot = False
        self.add_output_layer(label_size,
                              activation=tf.nn.softmax,
                              name="output")
        self.set_optimizer(tf.train.GradientDescentOptimizer)
        self.set_learning_rate(1e-1)

    def set_with_one_hot(self, with_one_hot):
        self.with_one_hot = with_one_hot
        return self

    def session(self, *args, **kwargs):
        return super(SoftmaxClassification, self) \
            .session(model=ModelSession,
                     label_size=self.label_size,
                     with_one_hot=self.with_one_hot,
                     *args, **kwargs)


class ModelSession(nn.AbstractModelSession):

    def __init__(self, *args, **kwargs):
        super(ModelSession, self).__init__(*args, **kwargs)
        self.label_size = kwargs["label_size"]
        self.with_one_hot = kwargs["with_one_hot"]
        self._Y = None
        self.accuracy = None
        self.validity = None

    def init_model(self):
        if self.with_one_hot:
            self.Y = tf.placeholder(tf.int32, [None, self.label_size])
            self._Y = tf.cast(self.Y, dtype=tf.float32)
        else:
            self.Y = tf.placeholder(tf.int32, [None, ])
            self._Y = tf.reshape(self.Y, [-1, 1])
            self._Y = tf.one_hot(self._Y, self.label_size)
            self._Y = tf.reshape(self._Y, [-1, self.label_size])

        self.pred = self.H
        self.cost = tf.reduce_mean(
            -tf.reduce_sum(self._Y * tf.log(self.H), axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(self.pred, axis=1), tf.argmax(self._Y, axis=1)),
            dtype=tf.float32))
        self.validity = tf.reduce_max(self.H, axis=1)

    def get_accuracy(self, x_data, y_data):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_data, self.Y: y_data})

    def get_validity(self, x_data, y_data):
        return self.sess.run(self.validity,
                             feed_dict={self.X: x_data, self.Y: y_data})
