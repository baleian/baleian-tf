from baleian.tf.nn.regression.linear_regression import *

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


def main():
    tf.set_random_seed(777)  # for reproducibility

    data = np.loadtxt(
        "data/linear_regression_test_data.csv",
        delimiter=",",
        dtype=np.float32
    )
    x_data = data[:, 0:-1]
    y_data = data[:, [-1]]
    test_x_data = x_data[:5]
    test_y_data = y_data[:5]

    net = LinearRegression(INPUT_SIZE, OUTPUT_SIZE)
    net.add_hidden_layer(128, name="hidden1")
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
        cost = model.get_cost(test_x_data, test_y_data)
        for x, y, p in zip(test_x_data, test_y_data, pred):
            print(x, y, p)
        print("cost: %f" % cost)


if __name__ == "__main__":
    main()
