
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import Agent


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

            self.inputs = tf.placeholder(tf.float32, [None, *state_size],
                                         name='Input_state')
            batch_size = tf.shape(self.inputs)[0]

            with tf.variable_scope('Convolutional_Layers'):
                self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                         inputs=self.inputs,
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
                self.conv4 = slim.conv2d(activation_fn=tf.nn.elu,
                                         inputs=self.conv3,
                                         num_outputs=32,
                                         kernel_size=[7, 7],
                                         stride=[1, 1],
                                         padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv4), 256,
                                          activation_fn=tf.nn.elu)

            with tf.variable_scope('LSTM'):
                # New LSTM Network with 256 cells
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256)
                c_size = lstm_cell.state_size.c
                h_size = lstm_cell.state_size.h

                # Initial state
                c_init = np.zeros((1, c_size), np.float32)
                h_init = np.zeros((1, h_size), np.float32)
                self.lstm_state_init = [c_init, h_init]

                # Input state
                c_in = tf.placeholder(tf.float32, [1, c_size])
                h_in = tf.placeholder(tf.float32, [1, h_size])
                self.state_in = (c_in, h_in)

                # tf.nn.dynamic_rnn expects inputs of shape
                # [batch_size, time, features], but the shape of hidden is
                # [batch_size, features]. We want the batch_size dimension to be
                # treated as the time dimension, so the input is redundantly
                # expanded to [1, batch_size, features].
                # The LSTM layer will assume it has 1 batch with a time
                # dimension of length batch_size.
                lstm_input = tf.expand_dims(hidden, [0])
                # [:1] is a trick to correctly get the dynamic shape.
                step_size = tf.shape(self.inputs)[:1]
                state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)

                # LSTM Output
                lstm_output, lstm_state = tf.nn.dynamic_rnn(lstm_cell,
                                                            lstm_input,
                                                            state_in,
                                                            step_size)
                lstm_c, lstm_h = lstm_state
                self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
                lstm_output = tf.reshape(lstm_output, [-1, 256])

            # Policy estimation
            self.policy = slim.fully_connected(
                lstm_output, action_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)

            # Value estimation
            self.value = slim.fully_connected(
                lstm_output, 1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)

        if scope != 'global':
            self.actions = tf.placeholder(tf.int32, [None], 'Action')
            self.actions_onehot = tf.onehot(self.actions,
                                            self.action_size,
                                            dtype=tf.float32)
            self.advantage = tf.placeholder(tf.float32, [None], 'Advantage')
            self.discounted_reward = tf.placeholder(tf.float32, [None],
                                                    'Discounted_Reward')
            self.responsible_outputs = tf.reduce_sum(
                self.policy * self.actions_onehot, [1])

            # Estimate the policy loss and regularize it by adding uncertainty
            # (subtracting entropy)
        	self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) *
        									  self.advantage)
        	self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))

	        # Estimate the value loss using the sum of squared errors.
	        self.value_loss = tf.nn.l2_loss(self.value - self.discounted_reward)

        	# Estimate the final loss.
        	self.loss = policy_loss + \
        				parameters.VALUE_REG * value_loss - \
        				parameters.ENTROPY_REG * entropy

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
