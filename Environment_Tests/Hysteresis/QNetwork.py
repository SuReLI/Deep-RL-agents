
import tensorflow as tf

from NetworkArchitecture import NetworkArchitecture

import parameters


class QNetwork:

    def __init__(self, state_size, action_size, scope):

        with tf.variable_scope(scope):
            self.state_size = state_size
            self.action_size = action_size

            # Define the model
            self.model = NetworkArchitecture(self.state_size, self.action_size)

            # Convolution network
            self.inputs = self.model.build_model()

            # Dueling DQN
            self.value, self.advantage = self.model.dueling()

            # Aggregation of value and advantage to get the Q-value
            adv_mean = tf.reduce_mean(self.advantage, axis=1, keep_dims=True)
            self.Qvalues = self.value + tf.subtract(self.advantage, adv_mean)

            self.predict = tf.argmax(self.Qvalues, 1)

            # Loss
            self.Qtarget = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions,
                                             self.action_size,
                                             dtype=tf.float32)

            self.Qaction = tf.reduce_sum(
                tf.multiply(self.Qvalues, self.actions_onehot), axis=1)

            self.td_error = tf.square(self.Qtarget - self.Qaction)
            self.loss = tf.reduce_mean(self.td_error)

            self.learning_rate = tf.placeholder(shape=None, dtype=tf.float32)

            self.trainer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate)
            self.train = self.trainer.minimize(self.loss)
