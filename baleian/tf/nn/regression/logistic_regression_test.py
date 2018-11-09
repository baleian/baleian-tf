from baleian.tf.nn.regression.logistic_regression import *

import tensorflow as tf
import numpy as np

# Set log level of tensorflow "WARN"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


INPUT_SIZE = 8
LEARNING_RATE = 1e-1
TRAINING_EPOCH = 1000


class LogisticRegressionTest(tf.test.TestCase):

    def test_logistic_regression(self):
        tf.set_random_seed(666)  # for reproducibility

        x_data = np.loadtxt(
            "data/logistic_regression_test_data.csv",
            delimiter=",",
            usecols=range(INPUT_SIZE),
            dtype=np.float32
        )
        y_data = np.loadtxt(
            "data/logistic_regression_test_data.csv",
            delimiter=",",
            usecols=[INPUT_SIZE],
            dtype=np.int32
        )

        net = LogisticRegression(INPUT_SIZE)
        net.add_hidden_layer(128, layer=tf.layers.Dense, activation=tf.nn.relu)
        net.add_hidden_layer(64, layer=tf.layers.Dense, activation=tf.nn.relu)
        net.set_learning_rate(LEARNING_RATE)

        with self.session() as sess:
            model = net.session(sess, name="net")
            accuracy = model.get_accuracy(x_data, y_data)
            self.assertEqual(int(accuracy * 100), 42)
            for epoch in range(TRAINING_EPOCH):
                model.training(x_data, y_data)
            accuracy = model.get_accuracy(x_data, y_data)
            self.assertEqual(int(accuracy * 100), 78)
