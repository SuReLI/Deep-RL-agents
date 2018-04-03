
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def load_mnist(data_dir):
    print("Loading dataset...")

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])
    train_X = tf.image.resize_images(mnist.train.images[:10000], (64, 64)).eval()
    train_X = (train_X - 0.5) / 0.5

    print("Dataset loaded !")
    return train_X
