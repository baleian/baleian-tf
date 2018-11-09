from baleian.tf.nn.classification.softmax_classification import *

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials import mnist

# Set log level of tensorflow "WARN"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


INPUT_SIZE = 784
LABEL_SIZE = 10
LEARNING_RATE = 1e-1
TRAINING_EPOCH = 15
TRAINING_BATCH = 100


def main():
    tf.set_random_seed(777)    # for reproducibility

    dataset = mnist.input_data.read_data_sets("data/mnist/", one_hot=False)
    iterations = int(dataset.train.num_examples / TRAINING_BATCH)

    net = SoftmaxClassification(INPUT_SIZE, LABEL_SIZE)
    net.add_hidden_layer(32)
    net.add_hidden_layer(16)
    net.set_learning_rate(LEARNING_RATE)
    net.set_with_one_hot(False)

    with tf.Session() as sess:
        model = net.session(sess, name="net")

        for epoch in range(TRAINING_EPOCH):
            avg_cost = 0
            for step in range(iterations):
                x_data, y_data = dataset.train.next_batch(TRAINING_BATCH)
                model.training(x_data, y_data)
                cost = model.get_cost(x_data, y_data)
                avg_cost += cost / iterations
            print("Epoch: %03d cost=%.9f" % (epoch + 1, avg_cost))

        test_x_data = dataset.test.images
        test_y_data = dataset.test.labels

        accuracy = model.get_accuracy(test_x_data, test_y_data)
        print("Accuracy: ", accuracy)

        pred = model.prediction(test_x_data)
        miss_matched = []
        for x, y, p in zip(test_x_data, test_y_data, pred):
            if (y != np.argmax(p)):
                miss_matched.append((x, y, p))
        print("Miss match count: ", len(miss_matched))

        print("Show miss matched case image")
        image, label, pred = random.choice(miss_matched)
        answer = int(np.argmax(pred))
        print("Label: %d (%.4f%%)" % (label, pred[label] * 100))
        print("Answer: %d (%.4f%%)"% (answer, pred[answer] * 100))
        print("Prediction: ", pred)
        plt.imshow(
            image.reshape(28, 28),
            cmap='Greys',
            interpolation='nearest')
        plt.show()


if __name__ == "__main__":
    main()
