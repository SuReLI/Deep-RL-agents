
import sys
import os

from multiprocessing import cpu_count
import argparse

import numpy as np
import tensorflow as tf

from Model import DCGAN
from utils import load_mnist


parser = argparse.ArgumentParser(description='Run DCGAN on MNist')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=100, help='Number of images in batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=25, help='Number of epoch')
parser.add_argument('--nb_gpu', dest='nb_gpu', type=int, default=1, help='Number of gpus')

args = parser.parse_args()

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, range(args.nb_gpu)))

    config = tf.ConfigProto(log_device_placement=False,
                            device_count={"CPU":cpu_count()-1, "GPU":args.nb_gpu})

    sess = tf.InteractiveSession(config=config)

    X = load_mnist("./data")

    GAN = DCGAN(sess, learning_rate=2e-4, batch_size=args.batch_size, nb_gpu=args.nb_gpu)

    sess.run(tf.global_variables_initializer())

    try:
        GAN.train(X, epochs=args.epoch)
    except KeyboardInterrupt:
        pass
