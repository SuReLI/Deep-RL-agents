
import tensorflow as tf

from settings import Settings


def build_model(inputs, trainable, scope):

    with tf.variable_scope(scope):
        if Settings.CONV:
            with tf.variable_scope('Convolutional_Layers'):
                conv1 = tf.layers.conv2d(inputs=inputs,
                                         filters=32,
                                         kernel_size=[8, 8],
                                         strides=[4, 4],
                                         padding='valid',
                                         activation=tf.nn.relu,
                                         trainable=trainable)
                conv2 = tf.layers.conv2d(conv1, 64, [4, 4], [2, 2], 'valid',
                                         activation=tf.nn.relu,
                                         trainable=trainable)
                conv3 = tf.layers.conv2d(conv2, 64, [3, 3], [1, 1], 'valid',
                                         activation=tf.nn.relu,
                                         trainable=trainable)

            # Flatten the output
            hidden = tf.layers.flatten(conv3)

        else:
            hidden = tf.layers.dense(inputs, 64, tf.nn.relu,
                                     name='hidden1', trainable=trainable)
            hidden = tf.layers.dense(hidden, 64, tf.nn.relu,
                                     name='hidden2', trainable=trainable)

        output = []
        for i in range(Settings.ACTION_SIZE):
            output.append(tf.layers.dense(hidden, Settings.NB_ATOMS,
                                          activation=tf.nn.softmax,
                                          name='hidden3_' + str(i + 1),
                                          trainable=trainable))
        return tf.stack(output, axis=1)


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
