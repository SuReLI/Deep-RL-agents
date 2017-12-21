
import tensorflow as tf
import numpy as np

from NetworkArchitecture import NetworkArchitecture
import settings


class Network:

    def __init__(self, state_size, action_size, scope):
        if scope == 'global':
            print("Initialization of the global network")

        with tf.variable_scope(scope):
            self.state_size = state_size
            self.action_size = action_size

            self.model = NetworkArchitecture(self.state_size)

            # Convolution network - or not
            if settings.CONV:
                self.inputs = self.model.build_conv()

            else:
                self.inputs = self.model.build_regular_layers()

            # LSTM Network - or not
            if settings.LSTM:
                # Input placeholder
                self.state_in = self.model.build_lstm()

                self.lstm_state_init = self.model.lstm_state_init
                self.state_out, model_output = self.model.return_output(True)

            else:
                model_output = self.model.return_output(False)

            # Policy estimation
            self.policy = tf.layers.dense(model_output, action_size,
                                          activation=tf.nn.softmax)

            # Value estimation
            self.value = tf.layers.dense(model_output, 1, activation=None)

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
            self.responsible_outputs = tf.clip_by_value(
                self.responsible_outputs, 1e-20, 1)

            # Estimate the policy loss and regularize it by adding uncertainty
            # (subtracting entropy)
            self.policy_loss = -tf.reduce_sum(tf.multiply(
                tf.log(self.responsible_outputs), self.advantages))
            self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))

            # Estimate the value loss using the sum of squared errors.
            self.value_loss = tf.reduce_sum(tf.square(self.advantages))
            # tf.reshape(self.value, [-1]) - self.discounted_reward))

            # Estimate the final loss.
            self.loss = self.policy_loss + \
                settings.VALUE_REG * self.value_loss - \
                settings.ENTROPY_REG * self.entropy

            # Fetch and clip the gradients of the local network.
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope)
            gradients = tf.gradients(self.loss, local_vars)
            clipped_gradients, self.grad_norm = tf.clip_by_global_norm(
                gradients, settings.MAX_GRADIENT_NORM)

            # Apply gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            'global')
            optimizer = tf.train.AdamOptimizer(settings.LEARNING_RATE)
            grads_and_vars = zip(clipped_gradients, global_vars)
            self.apply_grads = optimizer.apply_gradients(grads_and_vars)
