
import tensorflow as tf
import settings


class NetworkArchitecture:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def build_model(self):

        self.inputs = tf.placeholder(tf.float32, [None, *self.state_size],
                                     name='Input_state')

        if settings.CONV:

            with tf.variable_scope('Convolutional_Layers'):

                self.conv1 = tf.layers.conv2d(inputs=self.inputs,
                                              filters=32,
                                              kernel_size=[8, 8],
                                              stride=[4, 4],
                                              padding='VALID',
                                              activation=tf.nn.relu)
                self.conv2 = tf.layers.conv2d(inputs=self.conv1,
                                              filters=64,
                                              kernel_size=[4, 4],
                                              stride=[2, 2],
                                              padding='VALID',
                                              activation=tf.nn.relu)
                self.conv3 = tf.layers.conv2d(inputs=self.conv2,
                                              filters=64,
                                              kernel_size=[3, 3],
                                              stride=[1, 1],
                                              padding='VALID',
                                              activation=tf.nn.relu)

            # Flatten the output
            self.hidden = tf.layers.flatten(self.conv3)

        else:
            self.hidden = tf.layers.dense(self.inputs, 64,
                                          activation=tf.nn.relu)

        return self.inputs

    def dueling(self):

        self.adv_stream = tf.layers.dense(self.hidden, 32,
                                          activation=tf.nn.relu)
        self.value_stream = tf.layers.dense(self.hidden, 32,
                                            activation=tf.nn.relu)

        self.advantage = tf.layers.dense(self.adv_stream, self.action_size)
        self.value = tf.layers.dense(self.value_stream, 1)

        return self.value, self.advantage
