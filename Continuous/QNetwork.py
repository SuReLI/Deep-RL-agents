
import tensorflow as tf

from NetworkArchitecture import NetworkArchitecture

import parameters


class QNetwork:

    def __init__(self, state_size, action_size, bound, scope):

        with tf.variable_scope(scope):
            self.state_size = state_size
            self.action_size = action_size

            # Define the model
            self.model = NetworkArchitecture(self.state_size, self.action_size)

            # Define input layer
            self.inputs_state, self.inputs_action = self.model.get_input()

            # Dueling DQN
            self.actor_output = self.model.build_actor()
            self.predict_action = tf.mul(self.actor_output, bound)
            self.Qvalue = self.model.build_critic()

            # Critic Loss
            self.Qtarget = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions_onehot = tf.one_hot(self.actions,
                                             self.action_size,
                                             dtype=tf.float32)

            self.td_error = tf.square(self.Qtarget - self.Qvalue)
            self.loss = tf.reduce_mean(self.td_error)
            self.trainer_critic = tf.train.AdamOptimizer(
                learning_rate=parameters.LEARNING_RATE)
            self.train_critic = self.trainer.minimize(self.loss)

            self.action_grads = tf.gradients(self.Qvalue, self.actions)

            # Actor Loss
            self.vars = tf.trainable_variables()
            self.actor_grads = tf.gradients(
                self.predict_action, self.vars, -self.action_grads)

            grads_vars = zip(self.actor_grads, self.vars)
            self.trainer_actor = tf.train.AdamOptimizer(
                learning_rate=parameters.LEARNING_RATE)
            self.train_actor = self.trainer_actor.apply_gradients(grads_vars)
