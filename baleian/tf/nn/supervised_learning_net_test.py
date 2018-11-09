from baleian.tf.nn.optimizable_net import OptimizableNetwork
from baleian.tf.nn.supervised_learning_net import AbstractModelSession

import tensorflow as tf
import numpy as np

# Set log level of tensorflow "WARN"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class LinearRegressionModel(AbstractModelSession):

    def init_model(self):
        self.pred = self.H
        self.cost = tf.reduce_mean(tf.square(self.pred - self.Y))


INPUT_SIZE = 3
OUTPUT_SIZE = 1
LEARNING_RATE = 1e-6
TRAINING_EPOCH = 200


class SupervisedLearningTest(tf.test.TestCase):

    def test_linear_regression(self):
        tf.set_random_seed(777)  # for reproducibility

        data = np.loadtxt(
            "data/linear_regression_test_data.csv",
            delimiter=",",
            dtype=np.float32
        )
        x_data = data[:, 0:-1]
        y_data = data[:, [-1]]

        net = OptimizableNetwork(INPUT_SIZE)
        net.add_hidden_layer(128, layer=tf.layers.Dense, name="hidden1")
        net.add_output_layer(OUTPUT_SIZE)
        net.set_optimizer(tf.train.GradientDescentOptimizer)
        net.set_learning_rate(LEARNING_RATE)

        with self.session() as sess:
            model = net.session(sess, name="net", model=LinearRegressionModel)
            cost = model.get_cost(x_data, y_data)
            self.assertEqual(int(cost), 26139)
            for epoch in range(TRAINING_EPOCH):
                model.training(x_data, y_data)
            cost = model.get_cost(x_data, y_data)
            self.assertEqual(int(cost), 11)
