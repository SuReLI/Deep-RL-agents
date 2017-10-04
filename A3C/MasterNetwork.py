
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from NetworkArchitecture import NetworkArchitecture
import parameters


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class Network:

    def __init__(self, state_size, action_size, scope):
        if scope == 'global':
            print("Initialization of the global network")

        with tf.variable_scope(scope):
            self.state_size = state_size
            self.action_size = action_size

            self.model = NetworkArchitecture(self.state_size)

            if parameters.CONV:
                self.inputs = self.model.build_conv()

            else:
                self.inputs = self.model.build_regular_layers([32])

            if parameters.LSTM:
                # Input placeholder
                self.state_in = self.model.build_lstm()

                self.lstm_state_init = self.model.lstm_state_init
                self.state_out, model_output = self.model.return_output(True)

            else:
                model_output = self.model.return_output(False)

            # Policy estimation
            self.policy = slim.fully_connected(
                model_output, action_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)

            # Value estimation
            self.value = slim.fully_connected(
                model_output, 1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)

        if scope != 'global':
            self.actions = tf.placeholder(tf.int32, [None], 'Action')
            self.actions_onehot = tf.one_hot(self.actions,
                                             self.action_size,
                                             dtype=tf.float32)
            self.advantage = tf.placeholder(tf.float32, [None], 'Advantage')
            self.discounted_reward = tf.placeholder(tf.float32, [None],
                                                    'Discounted_Reward')
            self.responsible_outputs = tf.reduce_sum(
                self.policy * self.actions_onehot, [1])

            # Estimate the policy loss and regularize it by adding uncertainty
            # (subtracting entropy)
            self.policy_loss = -tf.reduce_sum(
                tf.log(self.responsible_outputs) * self.advantage)
            self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))

            # Estimate the value loss using the sum of squared errors.
            self.value_loss = tf.nn.l2_loss(
                self.value - self.discounted_reward)

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
            grads_and_vars = zip(gradients, global_vars)
            self.apply_grads = optimizer.apply_gradients(grads_and_vars)
