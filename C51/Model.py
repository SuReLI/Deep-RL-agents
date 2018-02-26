
import tensorflow as tf

from settings import Settings


def build_model(states, trainable, scope):
    """
    Define a deep neural network in tensorflow. The architecture of the network
    is defined in the 'settings' file : it can have convolution layers or just
    regular fully-connected layers. The output of the network is a tensor of
    size (BATCH_SIZE, ACTION_SIZE, NB_ATOMS) which corresponds to the estimation
    of the Q-value distribution of the input state over a support of NB_ATOMS
    (usually 51) atoms for each possible actions.

    Args:
        states   : a tensorflow placeholder to be feeded to get the network output
        trainable: whether the network is to be trained (main network) or to
                    have frozen weights (target network)
        scope    : the name of the tensorflow scope
    """
    with tf.variable_scope(scope):
        if Settings.CONV:
            with tf.variable_scope('Convolutional_Layers'):
                conv1 = tf.layers.conv2d(inputs=states,
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
            hidden = tf.layers.dense(states, 64, tf.nn.relu,
                                     name='hidden1', trainable=trainable)
            hidden = tf.layers.dense(hidden, 64, tf.nn.relu,
                                     name='hidden2', trainable=trainable)

        # Distributional perspective : for each action, a fully-connected layer
        # with softmax activation predicts the Q-value distribution
        output = []
        for i in range(Settings.ACTION_SIZE):
            output.append(tf.layers.dense(hidden, Settings.NB_ATOMS,
                                          activation=tf.nn.softmax,
                                          name='hidden3_' + str(i + 1),
                                          trainable=trainable))
        return tf.stack(output, axis=1)


def get_vars(scope, trainable):
    """
    Return every tensorflow variables defined in a given scope. Used to get
    network weights (with trainable == True for main networks and
    trainable == False for target networks).
    """
    if trainable:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    else:
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def copy_vars(src_vars, dest_vars, tau, name='update'):
    """
    Copy the value of every variable in a list src_vars into every corresponding
    variable in a list dest_vars with the update rate tau.

    Args:
        src_vars : the list of the variables to copy
        dest_vars: the list of the variables to modify
        tau      : the update copy rate 
        name     : the name of the operation
    """
    update_dest = []
    for src_var, dest_var in zip(src_vars, dest_vars):

        # Check if src_var and dest_var represents the same weight or bias
        src_name, dest_name = src_var.name, dest_var.name
        assert src_name[src_name.find("/"):] == dest_name[dest_name.find("/"):]

        op = dest_var.assign(tau * src_var + (1 - tau) * dest_var)
        update_dest.append(op)
    return tf.group(*update_dest, name=name)
