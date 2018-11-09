from baleian.tf.nn.classification.softmax_classification import *

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set log level of tensorflow "WARN"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


INPUT_SIZE = 16
LABEL_SIZE = 7
LEARNING_RATE = 1e-1
TRAINING_EPOCH = 1000


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def main():
    tf.set_random_seed(777)    # for reproducibility

    data = np.loadtxt(
        "data/softmax_classification_test_data.csv",
        delimiter=",",
        dtype=np.int32
    )
    x_data = data[:, 0:-1]
    y_data = data[:, -1]
    test_x_data = x_data[:100]
    test_y_data = y_data[:100]

    net = SoftmaxClassification(INPUT_SIZE, LABEL_SIZE)
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

        # y_test_one_hot_data = sess.run(tf.one_hot(test_y_data, OUTPUT_SIZE))
        pred = model.prediction(test_x_data)
        cost = model.get_cost(test_x_data, test_y_data)
        validity = model.get_validity(test_x_data, test_y_data)
        for y, p, v in zip(test_y_data, pred, validity):
            # Print serious mistakes (when the validity is more than 70%)
            if y != np.argmax(p) and v > 0.7:
                print("Serious Mistake!!", y, p, v)
            # Print correct answers, but the validity is less than 50%
            if y == np.argmax(p) and v < 0.5:
                print("Lucky!!", y, p, v)
        print("cost: %f" % cost)
        accuracy = model.get_accuracy(test_x_data, test_y_data)
        print("accuracy: %f" % accuracy)


if __name__ == "__main__":
    main()
