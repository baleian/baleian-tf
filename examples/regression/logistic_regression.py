from baleian.tf.nn.regression.logistic_regression import *

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set log level of tensorflow "WARN"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


INPUT_SIZE = 8
LEARNING_RATE = 1e-2
TRAINING_EPOCH = 1000


def main():
    tf.set_random_seed(777)    # for reproducibility

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
    test_x_data = x_data[:100]
    test_y_data = y_data[:100]

    net = LogisticRegression(INPUT_SIZE)
    net.add_hidden_layer(128, layer=tf.layers.Dense, activation=tf.nn.relu)
    net.add_hidden_layer(64, layer=tf.layers.Dense, activation=tf.nn.relu)
    net.set_learning_rate(LEARNING_RATE)

    with tf.Session() as sess:
        model = net.session(sess, name="net")

        epoch_history = []
        cost_history = []

        for epoch in range(TRAINING_EPOCH):
            model.training(x_data, y_data)
            cost = model.get_cost(x_data, y_data)
            epoch_history.append(epoch)
            cost_history.append(cost)

        plt.plot(epoch_history, cost_history)
        plt.show()

        pred = model.prediction(test_x_data)
        validity = model.get_validity(test_x_data, test_y_data)
        for y, p, v in zip(test_y_data, pred, validity):
            # Print serious mistakes (when the validity is more than 70%)
            if y != p and v > 0.7:
                print("Serious Mistake!!", y, p, v)
            # Print correct answers, but the validity is less than 50%
            if y == p and v < 0.5:
                print("Lucky!!", y, p, v)
        accuracy = model.get_accuracy(test_x_data, test_y_data)
        print("accuracy: %f" % accuracy)


if __name__ == "__main__":
    main()
