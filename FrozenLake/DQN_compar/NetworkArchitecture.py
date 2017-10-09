
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import flatten


class NetworkArchitecture:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def build_model(self):

        self.inputs = tf.placeholder(tf.float32, [None, *self.state_size],
                                     name='Input_state')

        # Flatten the output
        self.flat_conv = slim.fully_connected(self.inputs,
                                              16,
                                              activation_fn=tf.nn.elu)
        return self.inputs

    def dueling(self):

        self.advantage_stream = slim.fully_connected(self.flat_conv,
                                                    4,
                                                     activation_fn=tf.nn.elu)
        self.value_stream = slim.fully_connected(self.flat_conv,
                                                 4,
                                                 activation_fn=tf.nn.elu)

        self.advantage = slim.fully_connected(self.advantage_stream,
                                              self.action_size,
                                              activation_fn=None)
        self.value = slim.fully_connected(self.value_stream,
                                          1,
                                          activation_fn=None)
        return self.value, self.advantage
