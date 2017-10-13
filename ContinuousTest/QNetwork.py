
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

            # Actor critic
            self.actor_output = self.model.build_actor()
            self.predict = tf.multiply(self.actor_output, bound)
            self.Qvalue = self.model.build_critic()

            if scope == 'main':
                # Critic Loss
                self.Qtarget = tf.placeholder(shape=[None], dtype=tf.float32)

                self.td_error = tf.square(self.Qtarget - self.Qvalue)
                self.loss = tf.reduce_mean(self.td_error)
                self.trainer_critic = tf.train.AdamOptimizer(
                    learning_rate=parameters.CRITIC_LEARNING_RATE)
                self.train_critic = self.trainer_critic.minimize(self.loss)

                self.action_grads = tf.gradients(self.Qvalue,
                                                 self.inputs_action)

                # Actor Loss
                self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope='main/Actor')
                self.actor_grads = tf.gradients(
                    self.predict, self.vars, -self.action_grads[0])

                grads_vars = zip(self.actor_grads, self.vars)
                self.trainer_actor = tf.train.AdamOptimizer(
                    learning_rate=parameters.ACTOR_LEARNING_RATE)
                self.train_actor = self.trainer_actor.apply_gradients(
                    grads_vars)
