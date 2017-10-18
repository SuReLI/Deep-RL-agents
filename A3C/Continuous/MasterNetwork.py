
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import parameters


class Network:

    def __init__(self, sess, state_size, action_size,
                 low_bound, high_bound, scope):
        if scope == 'global':
            print("Initialization of the global network")

        self.sess = sess

        with tf.variable_scope(scope):
            self.state_size = state_size
            self.action_size = action_size
            self.low_bound = low_bound
            self.high_bound = high_bound

            self.state_input = tf.placeholder(tf.float32,
                                              [None, *self.state_size],
                                              'State_input')
            self.action_input = tf.placeholder(tf.float32,
                                               [None, self.action_size],
                                               'Action_input')
            self.reward_input = tf.placeholder(tf.float32,
                                               [None],
                                               'Reward_input')
            self.next_state_input = tf.placeholder(tf.float32,
                                                   [None, *self.state_size],
                                                   'Next_state_input')
            self.is_terminal_input = tf.placeholder(tf.float32,
                                                    [None],
                                                    'Done_input')

            self.action = self.build_actor_network('actor')
            self.q_value = self.build_critic_network('critic')

            self.actor_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
            self.critic_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        if scope != 'global':

            # Critic training
            self.target_input = tf.placeholder(tf.float32,
                                               [None, 1],
                                               'Target_input')
            weight_decay = tf.add_n([parameters.CRITIC_REG * tf.nn.l2_loss(var)
                                     for var in self.critic_vars])

            self.loss = tf.reduce_mean(
                tf.square(self.target_input - self.q_value)) + weight_decay

            trainer = tf.train.AdamOptimizer(parameters.CRITIC_LEARNING_RATE)
            self.train_critic = trainer.minimize(self.loss)

            self.q_gradient = tf.gradients(self.q_value, self.action_input)

            self.gradient_input = tf.placeholder(tf.float32,
                                                 [None, self.action_size],
                                                 'Grad_input')

            # Actor training
            self.actor_gradient = tf.gradients(self.action, self.actor_vars,
                                               -1 * self.gradient_input)

            self.global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 'global')

            trainer = tf.train.AdamOptimizer(parameters.ACTOR_LEARNING_RATE)
            self.train_actor = trainer.apply_gradients(zip(self.actor_gradient,
                                                           self.global_vars))

    def build_actor_network(self, scope):

        with tf.variable_scope(scope):
            hidden = tf.layers.dense(self.state_input, 8,
                                     activation=tf.nn.relu,
                                     name='dense')

            hidden_2 = tf.layers.dense(hidden, 8,
                                       activation=tf.nn.relu,
                                       name='dense_1')

            hidden_3 = tf.layers.dense(hidden_2, 8,
                                       activation=tf.nn.relu,
                                       name='dense_2')

            actions_unscaled = tf.layers.dense(hidden_3, self.action_size,
                                               name='dense_3')
        # Bound the actions to the valid range
        valid_range = self.high_bound - self.low_bound
        action_output = self.low_bound + \
            tf.nn.sigmoid(actions_unscaled) * valid_range

        return action_output

    def build_critic_network(self, scope):

        with tf.variable_scope('critic'):
            state_action = tf.concat([self.state_input, self.action_input],
                                     axis=1)
            hidden = tf.layers.dense(state_action, 8,
                                     activation=tf.nn.relu,
                                     name='dense')
            hidden_2 = tf.layers.dense(hidden, 8,
                                       activation=tf.nn.relu,
                                       name='dense_1')
            hidden_3 = tf.layers.dense(hidden_2, 8,
                                       activation=tf.nn.relu,
                                       name='dense_2')
            q_value = tf.layers.dense(hidden_3, 1,
                                      name='dense_3')

        return q_value

    def get_action(self, state):
        return self.sess.run(self.action,
                             feed_dict={self.state_input: state})

    def critic_train(self, state, action, target):
        self.sess.run(self.train_critic,
                      feed_dict={self.state_input: state,
                                 self.action_input: action,
                                 self.target_input: target})

    def q_value(self, state, action):
        return self.sess.run(self.q_value,
                             feed_dict={self.state_input: state,
                                        self.action_input: action})

    def action_gradient(self, state, action):
        return self.sess.run(self.q_gradient,
                             feed_dict={self.state_input: state,
                                        self.action_input: action})

    def actor_train(self, state, gradient):
        self.sess.run(self.train_actor,
                      feed_dict={self.state_input: state,
                                 self.gradient_input: gradient})

    def train(self, state, action, reward, next_state, done):

        next_action = self.get_action(next_state)

        next_q_value = self.q_value(next_state, next_action)

        target = reward + parameters.DISCOUNT * next_q_value * done

        self.critic_train(state, action, target)

        planned_action = self.get_action(state)

        actor_gradient = self.action_gradient(state, planned_action)

        self.actor_train(state, actor_gradient)

    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save_network(self, time_step):
        print('Save actor-network...', time_step)
        self.saver.save(self.sess, 'saved_networks/',
                        global_step=time_step)
