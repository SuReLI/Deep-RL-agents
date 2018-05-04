
import numpy as np
import tensorflow as tf

import os
import gzip
import urllib.request

SOURCE_PATH = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_PATH = 'MNIST_data/train-images-idx3-ubyte'

def load_mnist(data_dir):
    print("Loading dataset...")

    if not os.path.exists(DATA_PATH):
        os.mkdir('MNIST_data')
        filepath, _ = urllib.request.urlretrieve(SOURCE_PATH, DATA_PATH+'.gz')
        with gzip.GzipFile(filepath, 'rb') as gzip_file, open(DATA_PATH, 'wb') as decompressed_file:
            decompressed_file.write(gzip_file.read())

    fd = open('MNIST_data/train-images-idx3-ubyte')
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    fd.close()
    images = loaded[16:].reshape((60000, 28, 28, 1)).astype(float)
    images /= 255.0

    train_X = tf.image.resize_images(images[:10000], (64, 64)).eval()
    train_X = (train_X - 0.5) / 0.5

    print("Dataset loaded !")
    return train_X
