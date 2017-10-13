
import tensorflow as tf
import tensorflow.contrib.slim as slim

import parameters


class Actor:

    def __init__(self, state_size, action_size, low_bound, high_bound, scope):

        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope(scope):

            self.inputs_state = tf.placeholder(tf.float32,
                                               [None, *self.state_size],
                                               name='Input_state')

            self.hidden_actor_1 = tf.dense(self.inputs_state,
                                           8,
                                           activation=tf.nn.relu,
                                           name='Hidden_actor_1')

            self.hidden_actor_2 = tf.dense(self.hidden_actor_1,
                                           8,
                                           activation=tf.nn.relu,
                                           name='Hidden_actor_2')

            self.hidden_actor_3 = tf.dense(self.hidden_actor_2,
                                           8,
                                           activation=tf.nn.relu,
                                           name='Hidden_actor_3')

            # Output a value between 0 and 1
            self.actor_output = tf.dense(self.hidden_actor_3,
                                         self.action_size,
                                         activation=tf.nn.sigmoid,
                                         name='Actor_output')

            # Scale the output to cover the bounds
            self.predict_action = low_bound + \
                self.actor_output * (high_bound - low_bound)

            if scope == "targetActor":
                self.predict_action = tf.stop_gradient(self.predict_action)

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope=scope)


class Critic:

    def __init__(self, state_size, action_size, scope):

        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope(scope):

            self.inputs_state = tf.placeholder(tf.float32,
                                               [None, *self.state_size],
                                               name='Input_state')

            self.inputs_action = tf.placeholder(tf.float32,
                                                [None, self.action_size],
                                                name='Input_action')

            if parameters.CONV:

                with tf.variable_scope('Convolutional_Layers'):
                    self.conv1 = slim.conv2d(activation=tf.nn.relu,
                                             inputs=self.inputs_state,
                                             num_outputs=32,
                                             kernel_size=[8, 8],
                                             stride=[4, 4],
                                             padding='VALID')
                    self.conv2 = slim.conv2d(activation=tf.nn.relu,
                                             inputs=self.conv1,
                                             num_outputs=64,
                                             kernel_size=[4, 4],
                                             stride=[2, 2],
                                             padding='VALID')
                    self.conv3 = slim.conv2d(activation=tf.nn.relu,
                                             inputs=self.conv2,
                                             num_outputs=64,
                                             kernel_size=[3, 3],
                                             stride=[1, 1],
                                             padding='VALID')

                # Flatten the output
                self.hidden_critic_1 = tf.dense(flatten(self.conv3),
                                                8,
                                                activation=tf.nn.relu)

            else:
                self.hidden_critic_1 = tf.dense(self.inputs_state,
                                                8,
                                                activation=tf.nn.relu,
                                                name='Hidden_critic_1')

            self.state_action_layer = tf.concat([self.hidden_critic_1,
                                                 self.inputs_action], axis=1)

            self.hidden_critic_2 = tf.dense(self.state_action_layer,
                                            8,
                                            activation=tf.nn.relu,
                                            name='Hidden_critic_2')

            self.hidden_critic_3 = tf.dense(self.hidden_critic_2,
                                            8,
                                            activation=tf.nn.relu,
                                            name='Hidden_critic_3')

            self.Qvalue = tf.dense(self.hidden_critic_3,
                                  1,
                                  activation=None,
                                  name='QValue')









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
