
import tensorflow as tf

import Model
from settings import Settings


class QNetwork:

    def __init__(self, scope):
        print("Creation of the {} QNetwork...".format(scope))

        with tf.variable_scope(scope):

            # Input placeholder
            self.inputs = tf.placeholder(tf.float32,
                                         [None, *Settings.STATE_SIZE],
                                         name='Input_state')
            
            # Convolution network
            hidden_layer = Model.build_model(self.inputs)

            # Dueling DQN
            self.value, self.advantage = Model.dueling(hidden_layer)

            # Aggregation of value and advantage to get the Q-value
            adv_mean = tf.reduce_mean(self.advantage, axis=1, keepdims=True)
            self.Qvalues = self.value + tf.subtract(self.advantage, adv_mean)

            self.predict = tf.argmax(self.Qvalues, 1)

            # Loss
            self.Qtarget = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions,
                                             Settings.ACTION_SIZE,
                                             dtype=tf.float32)

            self.Qaction = tf.reduce_sum(
                tf.multiply(self.Qvalues, self.actions_onehot), axis=1)

            self.td_error = tf.square(self.Qtarget - self.Qaction)
            self.loss = tf.reduce_mean(self.td_error)
            self.trainer = tf.train.AdamOptimizer(
                learning_rate=Settings.LEARNING_RATE)
            self.train = self.trainer.minimize(self.loss)

        self.vars = Model.get_vars(scope, 'target' not in scope)

        print("QNetwork created !")
