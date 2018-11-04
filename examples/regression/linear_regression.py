from baleian.tf.nn.neural_net import NeuralNetwork

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set log level of tensorflow "WARN"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


INPUT_SIZE = 3
OUTPUT_SIZE = 1
LEARNING_RATE = 1e-7
TRAINING_EPOCH = 1000


class LinearRegression(NeuralNetwork):

    def __init__(self, input_size, name=None):
        super(LinearRegression, self).__init__(input_size, name)
        self._cost = None
        self._train = None

    def _on_layer_generated(self):
        self._cost = tf.reduce_mean(tf.square(self._H - self._Y))
        self._train = tf.train.GradientDescentOptimizer(LEARNING_RATE) \
            .minimize(self._cost)

    def train(self, sess: tf.Session, x_data, y_data):
        _, cost = sess.run(
            [self._train, self._cost],
            feed_dict={self._X: x_data, self._Y: y_data}
        )
        return cost

    def test(self, sess: tf.Session, x_data, y_data):
        return sess.run(
            [self._pred, self._cost],
            feed_dict={self._X: x_data, self._Y: y_data}
        )


def main():
    tf.set_random_seed(777)    # for reproducibility

    data = np.loadtxt(
        "data/linear_regression_test_data.csv",
        delimiter=",",
        dtype=np.float32
    )
    x_data = data[:, 0:-1]
    y_data = data[:, [-1]]
    test_x_data = x_data[:5]
    test_y_data = y_data[:5]

    model = LinearRegression(input_size=INPUT_SIZE)
    model.add_dense_layer(output_size=OUTPUT_SIZE)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        epoch_history = []
        cost_history = []

        for epoch in range(TRAINING_EPOCH):
            cost = model.train(sess, x_data, y_data)
            epoch_history.append(epoch)
            cost_history.append(cost)

        plt.plot(epoch_history, cost_history)
        plt.show()

        pred, cost = model.test(sess, test_x_data, test_y_data)
        for x, y, p in zip(test_x_data, test_y_data, pred):
            print(x, y, p)
        print("cost: %f" % cost)


if __name__ == "__main__":
    main()
