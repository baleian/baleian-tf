from baleian.tf.nn.linear_regression import *

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

    net = LinearRegression(INPUT_SIZE, name="net") \
        .add_layer(OUTPUT_SIZE, name="output") \
        .set_learning_rate(LEARNING_RATE)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model = net.session(sess)

        epoch_history = []
        cost_history = []

        for epoch in range(TRAINING_EPOCH):
            cost = model.train(x_data, y_data)
            epoch_history.append(epoch)
            cost_history.append(cost)

        plt.plot(epoch_history, cost_history)
        plt.show()

        pred, cost = model.test(test_x_data, test_y_data)
        for x, y, p in zip(test_x_data, test_y_data, pred):
            print(x, y, p)
        print("cost: %f" % cost)


if __name__ == "__main__":
    main()
