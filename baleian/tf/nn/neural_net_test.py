from baleian.tf.nn.neural_net import *

import tensorflow as tf
import numpy as np

# Set log level of tensorflow "WARN"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


INPUT_SIZE = 5
HIDDEN1_SIZE = 16
HIDDEN2_SIZE = 8
OUTPUT_SIZE = 4


class NeuralNetworkTest(tf.test.TestCase):

    def test_no_layer_prediction(self):
        x_data = [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]]
        net = NeuralNetwork(INPUT_SIZE)

        with self.session() as sess:
            model = net.session(sess, name="net")
            pred = model.prediction(x_data)
            self.assertAllClose(pred, x_data)

    def test_add_multi_layers(self):
        net = NeuralNetwork(INPUT_SIZE)
        net.add_hidden_layer(HIDDEN1_SIZE, name="hidden1")
        net.add_hidden_layer(HIDDEN2_SIZE, name="hidden2")
        net.add_output_layer(OUTPUT_SIZE, name="output")

        with self.session() as sess:
            model = net.session(sess, name="net")
            self.assertEqual(model.X.shape[1], INPUT_SIZE)
            self.assertEqual(model.layers[0].shape[1], HIDDEN1_SIZE)
            self.assertEqual(model.layers[1].shape[1], HIDDEN2_SIZE)
            self.assertEqual(model.layers[2].shape[1], OUTPUT_SIZE)
            self.assertEqual(model.pred.shape[1], OUTPUT_SIZE)

            w = model.get_weight("hidden1")
            self.assertEqual(w.shape[0], INPUT_SIZE)
            self.assertEqual(w.shape[1], HIDDEN1_SIZE)
            w = model.get_weight("hidden2")
            self.assertEqual(w.shape[0], HIDDEN1_SIZE)
            self.assertEqual(w.shape[1], HIDDEN2_SIZE)
            w = model.get_weight("output")
            self.assertEqual(w.shape[0], HIDDEN2_SIZE)
            self.assertEqual(w.shape[1], OUTPUT_SIZE)

            w1 = model.get_weight("hidden1")
            w2 = sess.run(model._layers[0].kernel)
            self.assertAllClose(w1, w2)

    def test_duplicate_model(self):
        net = NeuralNetwork(INPUT_SIZE)
        net.add_hidden_layer(HIDDEN1_SIZE, name="hidden1")
        net.add_output_layer(OUTPUT_SIZE, name="output")

        with self.session() as sess:
            model1 = net.session(sess, name="net1")
            net.add_hidden_layer(HIDDEN2_SIZE, name="hidden2")
            model2 = net.session(sess, name="net2")

            w = model1.get_weight("output")
            self.assertEqual(w.shape[0], HIDDEN1_SIZE)
            self.assertEqual(w.shape[1], OUTPUT_SIZE)
            w = model2.get_weight("output")
            self.assertEqual(w.shape[0], HIDDEN2_SIZE)
            self.assertEqual(w.shape[1], OUTPUT_SIZE)


class CustomModelSession(ModelSession):

    def init_model(self):
        self.pred = 2 * self.H


class NeuralNetworkCustomModelTest(tf.test.TestCase):

    def test_custom_model_prediction(self):
        x_data = [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]]
        net = NeuralNetwork(INPUT_SIZE)

        with self.session() as sess:
            model = net.session(sess, name="net", model=CustomModelSession)
            pred = model.prediction(x_data)
            self.assertAllClose(pred, np.multiply(2, x_data))
