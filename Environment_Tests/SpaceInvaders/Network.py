# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from settings import ENTROPY_REG, ACTION_SIZE


class Network:

    def __init__(self, thread_index, device):

        self.device = device
        self.thread_index = thread_index
        self.build_layers()

    def build_loss(self):

        with tf.device(self.device):

            self.action = tf.placeholder("float", [None, ACTION_SIZE])
            self.reward = tf.placeholder("float", [None])
            self.td_error = tf.placeholder("float", [None])

            log_pi = tf.log(tf.clip_by_value(self.policy, 1e-20, 1.0))

            entropy = -tf.reduce_sum(self.policy * log_pi, reduction_indices=1)

            policy_loss = - tf.reduce_sum(tf.reduce_sum(
                tf.multiply(log_pi, self.action), reduction_indices=1) *
                self.td_error + entropy * ENTROPY_REG)

            value_loss = 0.5 * tf.nn.l2_loss(self.reward - self.value)

            self.total_loss = policy_loss + value_loss

    def copy_network(self, src_network, name=None):
        src_vars = src_network.get_vars()
        dst_vars = self.get_vars()

        copy_ops = []
        with tf.device(self.device):
            with tf.name_scope(name, "Network", []) as name:
                for(src_var, dst_var) in zip(src_vars, dst_vars):
                    copy_ops.append(tf.assign(dst_var, src_var))

                return tf.group(*copy_ops, name=name)

    # weight initialization based on muupan'state code
    # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
    def _get_weights_bias(self, weight_shape):
        input_channels = weight_shape[0]
        output_channels = weight_shape[1]
        d = 1.0 / np.sqrt(input_channels)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape,
                                               minval=-d,
                                               maxval=d))
        bias = tf.Variable(tf.random_uniform(bias_shape,
                                             minval=-d,
                                             maxval=d))
        return weight, bias

    def _get_convolution(self, weight_shape):
        w = weight_shape[0]
        h = weight_shape[1]
        input_channels = weight_shape[2]
        output_channels = weight_shape[3]
        d = 1.0 / np.sqrt(input_channels * w * h)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape,
                                               minval=-d,
                                               maxval=d))
        bias = tf.Variable(tf.random_uniform(bias_shape,
                                             minval=-d,
                                             maxval=d))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")

    def build_layers(self):

        scope_name = "net_" + str(self.thread_index)
        with tf.device(self.device), tf.variable_scope(scope_name) as scope:
            self.state = tf.placeholder("float", [None, 84, 84, 4])

            self.W_conv1, self.b_conv1 = self._get_convolution([8, 8, 4, 16])
            self.W_conv2, self.b_conv2 = self._get_convolution([4, 4, 16, 32])

            self.W_fc1, self.b_fc1 = self._get_weights_bias([2592, 256])

            # lstm
            self.lstm = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

            # weight for policy output layer
            self.W_fc2, self.b_fc2 = self._get_weights_bias([256, ACTION_SIZE])

            # weight for value output layer
            self.W_fc3, self.b_fc3 = self._get_weights_bias([256, 1])

            h_conv1 = tf.nn.relu(self._conv2d(self.state,  self.W_conv1, 4) +
                                 self.b_conv1)
            h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) +
                                 self.b_conv2)

            h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)
            h_fc1_reshaped = tf.reshape(h_fc1, [1, -1, 256])

            # place holder for LSTM unrolling time step size.
            self.step_size = tf.placeholder(tf.float32, [1])

            self.initial_lstm_state_c = tf.placeholder(tf.float32, [1, 256])
            self.initial_lstm_state_h = tf.placeholder(tf.float32, [1, 256])
            self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state_c,
                                                                    self.initial_lstm_state_h)

            lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                              h_fc1_reshaped,
                                                              initial_state=self.initial_lstm_state,
                                                              sequence_length=self.step_size,
                                                              time_major=False,
                                                              scope=scope)

            lstm_outputs = tf.reshape(lstm_outputs, [-1, 256])

            # policy (output)
            self.policy = tf.nn.softmax(tf.matmul(lstm_outputs, self.W_fc2) +
                                        self.b_fc2)

            # value (output)
            v_ = tf.matmul(lstm_outputs, self.W_fc3) + self.b_fc3
            self.value = tf.reshape(v_, [-1])

            scope.reuse_variables()
            self.W_lstm = tf.get_variable("basic_lstm_cell/kernel")
            self.b_lstm = tf.get_variable("basic_lstm_cell/bias")

            self.reset_state()

    def reset_state(self):
        self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]),
                                                            np.zeros([1, 256]))

    def run_policy_and_value(self, sess, state):
        feed_dict = {self.state: [state],
                     self.initial_lstm_state_c: self.lstm_state_out[0],
                     self.initial_lstm_state_h: self.lstm_state_out[1],
                     self.step_size: [1]}
        pi_out, v_out, self.lstm_state_out = sess.run([self.policy,
                                                       self.value,
                                                       self.lstm_state],
                                                      feed_dict=feed_dict)
        return (pi_out[0], v_out[0])

    def run_value(self, sess, state):
        prev_lstm_state_out = self.lstm_state_out
        v_out, _ = sess.run([self.value, self.lstm_state],
                            feed_dict={self.state: [state],
                                       self.initial_lstm_state_c: self.lstm_state_out[0],
                                       self.initial_lstm_state_h: self.lstm_state_out[1],
                                       self.step_size: [1]})

        # roll back lstm state
        self.lstm_state_out = prev_lstm_state_out
        return v_out[0]

    def get_vars(self):
        return [self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_fc1, self.b_fc1,
                self.W_lstm, self.b_lstm,
                self.W_fc2, self.b_fc2,
                self.W_fc3, self.b_fc3]
