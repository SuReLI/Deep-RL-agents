
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import flatten

import parameters


class NetworkArchitecture:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def build_model(self):

        self.inputs = tf.placeholder(tf.float32, [None, *self.state_size],
                                     name='Input_state')

        if parameters.CONV:

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
                self.conv3 = slim.conv2d(activation_fn=tf.nn.elu,
                                         inputs=self.conv2,
                                         num_outputs=64,
                                         kernel_size=[3, 3],
                                         stride=[1, 1],
                                         padding='VALID')

            # Flatten the output
            self.hidden = flatten(self.conv3)

        else:
            self.hidden = slim.fully_connected(self.inputs,
                                               64,
                                               activation_fn=tf.nn.elu)

        return self.inputs

    def dueling(self):

        self.advantage_stream = slim.fully_connected(self.hidden,
                                                     32,
                                                     activation_fn=tf.nn.elu)
        self.value_stream = slim.fully_connected(self.hidden,
                                                 32,
                                                 activation_fn=tf.nn.elu)

        self.advantage = slim.fully_connected(self.advantage_stream,
                                              self.action_size,
                                              activation_fn=None)
        self.value = slim.fully_connected(self.value_stream,
                                          1,
                                          activation_fn=None)
        return self.value, self.advantage
