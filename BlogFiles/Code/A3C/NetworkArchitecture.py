
import tensorflow as tf
import numpy as np

import parameters


class NetworkArchitecture:

    def __init__(self, state_size):
        self.state_size = state_size

    def build_regular_layers(self, activation_fn=tf.nn.relu):

        self.inputs = tf.placeholder(tf.float32, [None, *self.state_size],
                                     name='Input_state')

        layers = [self.inputs]

        size = parameters.LAYERS_SIZE
        for n in range(len(size)):
            layer = tf.layers.dense(
                layers[n], size[n], activation=activation_fn)
            layers.append(layer)

        self.hidden = layers[-1]
        return self.inputs

    def build_conv(self):
        """Define a succesion of convolutional layers followed by a fully
        connected layer and return the input placeholder"""

        # Placeholder for the input states (e.g the raw pixel frames)
        self.inputs = tf.placeholder(tf.float32, [None, *self.state_size],
                                     name='Input_state')

        with tf.variable_scope('Convolutional_Layers'):
            self.conv1 = tf.layers.conv2d(inputs=self.inputs,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding='valid',
                                          activation=tf.nn.relu)
            self.conv2 = tf.layers.conv2d(inputs=self.conv1,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding='valid',
                                          activation=tf.nn.relu)

        # Flatten the output
        flat_conv2 = tf.layers.flatten(self.conv2)
        self.hidden = tf.layers.dense(flat_conv2, 256, activation=tf.nn.elu)
        return self.inputs

    def build_lstm(self):

        with tf.variable_scope('LSTM'):
            # New LSTM Network with 256 cells
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(parameters.LSTM_CELLS)
            c_size = lstm_cell.state_size.c
            h_size = lstm_cell.state_size.h

            # Initial state
            c_init = np.zeros((1, c_size), np.float32)
            h_init = np.zeros((1, h_size), np.float32)
            self.lstm_state_init = [c_init, h_init]

            # Input state
            c_in = tf.placeholder(tf.float32, [1, c_size])
            h_in = tf.placeholder(tf.float32, [1, h_size])

            # tf.nn.dynamic_rnn expects inputs of shape
            # [batch_size, time, features], but the shape of hidden is
            # [batch_size, features]. We want the batch_size dimension to
            # be treated as the time dimension, so the input is redundantly
            # expanded to [1, batch_size, features].
            # The LSTM layer will assume it has 1 batch with a time
            # dimension of length batch_size.
            lstm_input = tf.expand_dims(self.hidden, [0])
            # [:1] is a trick to correctly get the dynamic shape.
            step_size = tf.shape(self.inputs)[:1]
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)

            # LSTM Output
            lstm_output, lstm_state = tf.nn.dynamic_rnn(lstm_cell,
                                                        lstm_input,
                                                        step_size,
                                                        state_in)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            self.output = tf.reshape(lstm_output, [-1, parameters.LSTM_CELLS])
            return (c_in, h_in)

    def return_output(self, lstm):

        if lstm:
            return self.state_out, self.output

        else:
            return self.hidden
