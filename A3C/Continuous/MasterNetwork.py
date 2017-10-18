
import tensorflow as tf
import numpy as np

import parameters


class Network:

    def __init__(self, state_size, action_size, low_bound, high_bound, scope):
        if scope == 'global':
            print("Initialization of the global network")

        with tf.variable_scope(scope):
            self.state_size = state_size
            self.action_size = action_size
            self.low_bound = low_bound
            self.high_bound = high_bound

            # placeholders
            self.state_ph = tf.placeholder(dtype=tf.float32,
                                           shape=[None, self.state_size])
            self.action_ph = tf.placeholder(dtype=tf.float32,
                                            shape=[None, self.action_size])
            self.reward_ph = tf.placeholder(dtype=tf.float32,
                                            shape=[None])
            self.next_state_ph = tf.placeholder(dtype=tf.float32,
                                                shape=[None, self.state_size])
            self.is_not_terminal_ph = tf.placeholder(dtype=tf.float32,
                                                     shape=[None])

            self.suggested_action = self.generate_actor_network(self.state_ph)
            self.value_of_suggested_action = self.generate_critic_network(
                self.state_ph, self.suggested_action)

            self.next_action = self.generate_actor_network(self.next_state_ph)

            self.value = self.generate_critic_network(self.state_ph,
                                                      self.action_ph)

        if scope != 'global':
            self.actions = tf.placeholder(tf.int32, [None], 'Action')
            self.actions_onehot = tf.one_hot(self.actions,
                                             self.action_size,
                                             dtype=tf.float32)
            self.advantages = tf.placeholder(tf.float32, [None], 'Advantage')
            self.discounted_reward = tf.placeholder(tf.float32, [None],
                                                    'Discounted_Reward')
            self.responsible_outputs = tf.reduce_sum(
                self.policy * self.actions_onehot, [1])

            # Estimate the policy loss and regularize it by adding uncertainty
            # (subtracting entropy)
            self.policy_loss = -tf.reduce_sum(
                tf.log(self.responsible_outputs) * self.advantages)
            self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))

            # Estimate the value loss using the sum of squared errors.
            self.value_loss = tf.reduce_sum(tf.square(
                tf.reshape(self.value, [-1]) - self.discounted_reward))

            # Estimate the final loss.
            self.loss = self.policy_loss + \
                parameters.VALUE_REG * self.value_loss - \
                parameters.ENTROPY_REG * self.entropy

            # Fetch and clip the gradients of the local network.
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope)
            gradients = tf.gradients(self.loss, local_vars)
            clipped_gradients, self.grad_norm = tf.clip_by_global_norm(
                gradients, parameters.MAX_GRADIENT_NORM)

            # Apply gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            'global')
            optimizer = tf.train.AdamOptimizer(parameters.LEARNING_RATE)
            grads_and_vars = zip(clipped_gradients, global_vars)
            self.apply_grads = optimizer.apply_gradients(grads_and_vars)

    # Actor definition :
    def generate_actor_network(self, states):
        hidden = tf.layers.dense(states, 8,
                                 activation=tf.nn.relu, name='dense')
        hidden_2 = tf.layers.dense(hidden, 8,
                                   activation=tf.nn.relu, name='dense_1')
        hidden_3 = tf.layers.dense(hidden_2, 8,
                                   activation=tf.nn.relu, name='dense_2')
        actions_unscaled = tf.layers.dense(hidden_3, self.action_size,
                                           name='dense_3')
        # bound the actions to the valid range
        valid_range = self.high_bound - self.low_bound
        actions = self.low_bound + \
            tf.nn.sigmoid(actions_unscaled) * valid_range
        return actions

    # Critic definition :
    def generate_critic_network(self, states, actions):
        state_action = tf.concat([states, actions], axis=1)
        hidden = tf.layers.dense(state_action, 8,
                                 activation=tf.nn.relu, name='dense')
        hidden_2 = tf.layers.dense(hidden, 8,
                                   activation=tf.nn.relu, name='dense_1')
        hidden_3 = tf.layers.dense(hidden_2, 8,
                                   activation=tf.nn.relu, name='dense_2')
        q_values = tf.layers.dense(hidden_3, 1,
                                   name='dense_3')
        return q_values
