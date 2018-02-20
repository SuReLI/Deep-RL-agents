
import tensorflow as tf
from settings import Settings


def build_model(inputs):

    if Settings.CONV:

        with tf.variable_scope('Convolutional_Layers'):

            conv1 = tf.layers.conv2d(inputs=inputs,
                                     filters=32,
                                     kernel_size=[8, 8],
                                     stride=[4, 4],
                                     activation=tf.nn.relu)

            conv2 = tf.layers.conv2d(conv1, 64, [4, 4], [2, 2],
                                     activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(conv2, 64, [3, 3], [1, 1],
                                     activation=tf.nn.relu)

        # Flatten the output
        hidden = tf.layers.flatten(conv3)

    else:
        hidden = tf.layers.dense(inputs, 64,
                                 activation=tf.nn.relu)

    return hidden


def dueling(hidden):

    adv_stream = tf.layers.dense(hidden, 32,
                                 activation=tf.nn.relu)
    value_stream = tf.layers.dense(hidden, 32,
                                   activation=tf.nn.relu)

    advantage = tf.layers.dense(adv_stream, Settings.NB_ATOMS * Settings.ACTION_SIZE)
    advantage = tf.reshape(advantage, [-1, Settings.ACTION_SIZE, Settings.NB_ATOMS])
    value = tf.layers.dense(value_stream, Settings.NB_ATOMS)
    value = tf.reshape(value, [-1, 1, Settings.NB_ATOMS])

    return value, advantage


def get_vars(scope, trainable):
    if trainable:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    else:
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def copy_vars(src_vars, dest_vars, tau, name):
    update_dest = []
    for src_var, dest_var in zip(src_vars, dest_vars):
        op = dest_var.assign(tau * src_var + (1 - tau) * dest_var)
        update_dest.append(op)
    return tf.group(*update_dest, name=name)
