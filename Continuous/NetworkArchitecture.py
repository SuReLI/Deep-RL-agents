
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import flatten

import parameters


class NetworkArchitecture:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def get_input(self):
        self.inputs_state = tf.placeholder(tf.float32,
                                           [None, *self.state_size],
                                           name='Input_state')
        self.inputs_action = tf.placeholder(tf.float32,
                                            [None, self.action_size],
                                            name='Input_action')
        return self.inputs_state, self.inputs_action

    def build_actor(self):

        with tf.variable_scope('Hidden_actor'):

            self.hidden_actor_1 = slim.fully_connected(self.inputs_state,
                                                       64,
                                                       activation_fn=tf.nn.elu)
            self.hidden_actor_2 = slim.fully_connected(self.hidden_actor_1,
                                                       64,
                                                       activation_fn=tf.nn.elu)
        self.actor = slim.fully_connected(self.hidden_actor_2,
                                          self.action_size,
                                          activation_fn=tf.tanh)
        return self.actor

    def build_critic(self):

        if parameters.CONV:

            with tf.variable_scope('Convolutional_Layers'):
                self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                         inputs=self.inputs_state,
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
            self.hidden_critic_1 = slim.fully_connected(flatten(self.conv3),
                                                        64,
                                                        activation_fn=tf.nn.elu)

        else:
            self.hidden_critic_1 = slim.fully_connected(self.inputs_state,
                                                        64,
                                                        activation_fn=tf.nn.elu)

        self.actor_layer = slim.fully_connected(self.inputs_action,
                                                64,
                                                activation_fn=tf.nn.elu)
        self.state_action_layer = self.hidden_critic_1 + self.actor_layer
        self.hidden_critic_2 = slim.fully_connected(self.state_action_layer,
                                                    64,
                                                    activation_fn=tf.nn.elu)

        self.value = slim.fully_connected(self.hidden_critic_2,
                                          1,
                                          activation_fn=None)
        return self.value
