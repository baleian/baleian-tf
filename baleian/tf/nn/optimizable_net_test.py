from baleian.tf.nn.optimizable_net import *

import tensorflow as tf

# Set log level of tensorflow "WARN"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


INPUT_SIZE = 5


class OptimizableModel(AbstractModelSession):

    def init_model(self):
        self.pred = self.H
        self.cost = tf.reduce_mean(tf.square(self.pred - self.X))

    def training(self, x_data):
        self.sess.run(self.optimize, feed_dict={self.X: x_data})

    def get_cost(self, x_data):
        return self.sess.run(self.cost, feed_dict={self.X: x_data})


class OptimizableNetworkTest(tf.test.TestCase):

    def test_if_weights_are_optimized(self):
        tf.set_random_seed(777)  # for reproducibility

        x_data = [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]]
        net = OptimizableNetwork(INPUT_SIZE)
        net.add_output_layer(INPUT_SIZE)
        net.set_optimizer(tf.train.GradientDescentOptimizer)
        net.set_learning_rate(1e-2)

        with self.session() as sess:
            model = net.session(sess, name="net", model=OptimizableModel)
            pred1 = model.prediction(x_data)
            pred2 = model.prediction(x_data)
            self.assertAllClose(pred1, pred2)

            cost1 = model.get_cost(x_data)
            model.training(x_data)
            cost2 = model.get_cost(x_data)
            self.assertLess(cost2, cost1)

            pred2 = model.prediction(x_data)
            self.assertNotAllClose(pred1, pred2)
            pred3 = model.prediction(x_data)
            self.assertAllClose(pred2, pred3)
