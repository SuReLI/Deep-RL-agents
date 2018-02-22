
import tensorflow as tf
from settings import Settings


def build_model(inputs, trainable, reuse, scope):

    with tf.variable_scope(scope):

        if Settings.CONV:
            conv1 = tf.layers.conv2d(inputs=inputs,
                                     filters=32,
                                     kernel_size=[8, 8],
                                     stride=[4, 4],
                                     activation=tf.nn.relu,
                                     trainable=trainable,
                                     reuse=reuse,
                                     name='conv_1')

            conv2 = tf.layers.conv2d(conv1, 64, [4, 4], [2, 2],
                                     activation=tf.nn.relu,
                                     trainable=trainable, reuse=reuse,
                                     name='conv_2')
            conv3 = tf.layers.conv2d(conv2, 64, [3, 3], [1, 1],
                                     activation=tf.nn.relu,
                                     trainable=trainable, reuse=reuse,
                                     name='conv_3')

            # Flatten the output
            hidden = tf.layers.flatten(conv3)

        else:
            hidden = tf.layers.dense(inputs, 64,
                                     trainable=trainable, reuse=reuse,
                                     activation=tf.nn.relu, name='dense')

        adv_stream = tf.layers.dense(hidden, 32,
                                     trainable=trainable, reuse=reuse,
                                     activation=tf.nn.relu, name='adv_stream')

        advantage = tf.layers.dense(adv_stream, Settings.NB_ATOMS * Settings.ACTION_SIZE,
                                    trainable=trainable, reuse=reuse, name='adv')
        advantage = tf.reshape(
            advantage, [-1, Settings.ACTION_SIZE, Settings.NB_ATOMS])

        advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)


        value_stream = tf.layers.dense(hidden, 32,
                                     trainable=trainable, reuse=reuse,
                                     activation=tf.nn.relu, name='value_stream')
        value = tf.layers.dense(value_stream, Settings.NB_ATOMS,
                                     trainable=trainable, reuse=reuse,
                                     activation=tf.nn.relu, name='value')
        value = tf.reshape(value, [-1, 1, Settings.NB_ATOMS])

        Qdistrib = tf.nn.softmax(value + advantage - advantage_mean, axis=2)

    return Qdistrib


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
