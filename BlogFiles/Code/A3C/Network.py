# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from settings import ENTROPY_REG, ACTION_SIZE


class Network:

    def __init__(self, thread_index, device):

        self.device = device
        self.thread_index = thread_index
        self.build_layers()

    def build_layers(self):

        scope_name = "net_" + str(self.thread_index)
        with tf.device(self.device), tf.variable_scope(scope_name) as scope:
            self.state = tf.placeholder("float", [None, 84, 84, 4])

            conv1 = tf.layers.conv2d(self.state, 64, [8, 8],
                                     strides=4, padding="valid",
                                     activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv1, 128, [4, 4],
                                     strides=2, padding="valid",
                                     activation=tf.nn.relu)

            conv2_flat = tf.layers.flatten(h_conv2)
            hidden = tf.layers.dense(conv2_flat, 256, activation=tf.nn.relu)
            hidden_reshaped = tf.reshape(h_fc1, [1, -1, 256])

            self.lstm = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

            # place holder for LSTM unrolling time step size.
            self.step_size = tf.placeholder(tf.float32, [1])

            self.initial_lstm_state_c = tf.placeholder(tf.float32, [1, 256])
            self.initial_lstm_state_h = tf.placeholder(tf.float32, [1, 256])
            self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state_c,
                                                                    self.initial_lstm_state_h)

            lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                              hidden_reshaped,
                                                              initial_state=self.initial_lstm_state,
                                                              sequence_length=self.step_size,
                                                              time_major=False,
                                                              scope=scope)

            lstm_outputs = tf.reshape(lstm_outputs, [-1, 256])

            # Policy estimation
            self.policy = tf.layers.dense(
                lstm_outputs, ACTION_SIZE, activation=tf.nn.softmax)

            # Value estimation
            v_ = tf.layers.dense(lstm_outputs, 1, activation=None)
            self.value = tf.reshape(v_, [-1])

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
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope="net_" + str(self.thread_index))
