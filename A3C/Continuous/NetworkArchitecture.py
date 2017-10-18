
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import flatten

import numpy as np

import parameters


class NetworkArchitecture:

    def __init__(self, state_size):
        self.state_size = state_size

    def build_regular_layers(self, activation_fn=tf.nn.elu):

        self.inputs = tf.placeholder(tf.float32, [None, *self.state_size],
                                     name='Input_state')

        layers = [self.inputs]

        size = parameters.LAYERS_SIZE
        for n in range(len(size)):
            layer = slim.fully_connected(inputs=layers[n],
                                         num_outputs=size[n],
                                         activation_fn=activation_fn)
            layers.append(layer)

        self.hidden = layers[-1]
        return self.inputs

    def build_conv(self):

        self.inputs = tf.placeholder(tf.float32, [None, *self.state_size],
                                     name='Input_state')

        with tf.variable_scope('Convolutional_Layers'):
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.inputs,
                                     num_outputs=32,
                                     kernel_size=[8, 8],
                                     stride=[4, 4],
                                     padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv1,
                                     num_outputs=64,
                                     kernel_size=[4, 4],
                                     stride=[2, 2],
                                     padding='VALID')

        # Flatten the output
        flat_conv2 = flatten(self.conv2)
        self.hidden = slim.fully_connected(flat_conv2, 256,
                                           activation_fn=tf.nn.elu)
        return self.inputs

    def return_output(self):
        return self.hidden
