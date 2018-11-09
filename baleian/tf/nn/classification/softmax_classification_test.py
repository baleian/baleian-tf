from baleian.tf.nn.classification.softmax_classification import *

import tensorflow as tf
import numpy as np

# Set log level of tensorflow "WARN"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


INPUT_SIZE = 16
LABEL_SIZE = 7
LEARNING_RATE = 1e-1
TRAINING_EPOCH = 1000


class SoftmaxClassificationTest(tf.test.TestCase):

    def test_softmax_classification(self):
        tf.set_random_seed(666)  # for reproducibility

        data = np.loadtxt(
            "data/softmax_classification_test_data.csv",
            delimiter=",",
            dtype=np.int32
        )
        x_data = data[:, 0:-1]
        y_data = data[:, -1]

        net = SoftmaxClassification(INPUT_SIZE, LABEL_SIZE)
        net.set_learning_rate(LEARNING_RATE)

        # test softmax classification with argmax
        net.set_with_one_hot(False)
        with self.session() as sess:
            model = net.session(sess, name="net1")
            accuracy = model.get_accuracy(x_data, y_data)
            self.assertEqual(int(accuracy * 100), 7)
            for epoch in range(TRAINING_EPOCH):
                model.training(x_data, y_data)
            accuracy = model.get_accuracy(x_data, y_data)
            self.assertEqual(int(accuracy * 100), 100)

            pred = model.prediction(x_data)
            correct = 0
            for y, p in zip(y_data, pred):
                if y == np.argmax(p):
                    correct += 1
            accuracy2 = correct / len(y_data)
            self.assertEqual(int(accuracy * 100), int(accuracy2 * 100))

        # test softmax classification without argmax
        eye = np.eye(LABEL_SIZE)
        y_data = [eye[y] for y in y_data]
        net.set_with_one_hot(True)
        with self.session() as sess:
            model = net.session(sess, name="net2")
            accuracy = model.get_accuracy(x_data, y_data)
            self.assertEqual(int(accuracy * 100), 6)
            for epoch in range(TRAINING_EPOCH):
                model.training(x_data, y_data)
            accuracy = model.get_accuracy(x_data, y_data)
            self.assertEqual(int(accuracy * 100), 100)

            pred = model.prediction(x_data)
            correct = 0
            for y, p in zip(y_data, pred):
                if np.argmax(y) == np.argmax(p):
                    correct += 1
            accuracy2 = correct / len(y_data)
            self.assertEqual(int(accuracy * 100), int(accuracy2 * 100))
