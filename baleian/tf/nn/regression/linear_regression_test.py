from baleian.tf.nn.regression.linear_regression import *

import tensorflow as tf
import numpy as np

# Set log level of tensorflow "WARN"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


INPUT_SIZE = 3
OUTPUT_SIZE = 1
TRAINING_EPOCH = 200


class LinearRegressionTest(tf.test.TestCase):

    def test_linear_regression(self):
        tf.set_random_seed(666)  # for reproducibility

        data = np.loadtxt(
            "data/linear_regression_test_data.csv",
            delimiter=",",
            dtype=np.float32
        )
        x_data = data[:, 0:-1]
        y_data = data[:, [-1]]

        # test_linear_regression_default
        net = LinearRegression(INPUT_SIZE, OUTPUT_SIZE)
        with self.session() as sess:
            model1 = net.session(sess, name="net1")
            cost = model1.get_cost(x_data, y_data)
            self.assertEqual(int(cost), 58836)
            for epoch in range(TRAINING_EPOCH):
                model1.training(x_data, y_data)
            cost = model1.get_cost(x_data, y_data)
            self.assertEqual(int(cost), 12)

        # test_linear_regression_custom
        net.set_learning_rate(1e-10)
        with self.session() as sess:
            model2 = net.session(sess, name="net2")
            cost = model2.get_cost(x_data, y_data)
            print(cost)
            self.assertEqual(int(cost), 46366)
            for epoch in range(int(TRAINING_EPOCH / 2)):
                model2.training(x_data, y_data)
            cost = model2.get_cost(x_data, y_data)
            self.assertEqual(int(cost), 46330)
            for epoch in range(int(TRAINING_EPOCH / 2)):
                model2.training(x_data, y_data)
            cost = model2.get_cost(x_data, y_data)
            self.assertEqual(int(cost), 46293)

        # test_change_learning_rate_in_the_middle_of_run
        net = LinearRegression(INPUT_SIZE, OUTPUT_SIZE)
        net.set_learning_rate(1e-10)
        with self.session() as sess:
            model3 = net.session(sess, name="net3")
            cost = model3.get_cost(x_data, y_data)
            self.assertEqual(int(cost), 54714)
            for epoch in range(int(TRAINING_EPOCH / 2)):
                model3.training(x_data, y_data)
            cost = model3.get_cost(x_data, y_data)
            self.assertEqual(int(cost), 54671)
            model3.learning_rate = 1e-6
            for epoch in range(int(TRAINING_EPOCH / 2)):
                model3.training(x_data, y_data)
            cost = model3.get_cost(x_data, y_data)
            self.assertEqual(int(cost), 68)
