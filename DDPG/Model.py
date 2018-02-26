
import tensorflow as tf

from settings import Settings


def build_actor(states, trainable, scope):
    """
    Define an actor network that predicts the best continuous action to perform
    given the current state of an environment.

    Args:
        states   : a tensorflow placeholder to be feeded to get the network output
        trainable: whether the network is to be trained (main network) or to
                    have frozen weights (target network)
        scope    : the name of the tensorflow scope
    """
    with tf.variable_scope(scope):

        layer = states

        # Convolution layers
        if hasattr(Settings, 'CONV_LAYERS') and Settings.CONV_LAYERS:
            for i, layer_settings in enumerate(Settings.CONV_LAYERS):
                layer = tf.layers.conv2d(inputs=layer,
                                         activation=tf.nn.relu,
                                         trainable=trainable,
                                         name='conv_'+str(i),
                                         **layer_settings)

            layer = tf.layers.flatten(layer)

        # Fully connected layers
        for i, nb_neurons in enumerate(Settings.HIDDEN_ACTOR_LAYERS):
            layer = tf.layers.dense(layer, nb_neurons,
                                    trainable=trainable,
                                    activation=tf.nn.relu,
                                    name='dense_'+str(i))

        actions_unscaled = tf.layers.dense(layer, Settings.ACTION_SIZE,
                                           trainable=trainable,
                                           name='dense_last')
        # Bound the actions to the valid range
        valid_range = Settings.HIGH_BOUND - Settings.LOW_BOUND
        actions = Settings.LOW_BOUND + tf.nn.sigmoid(actions_unscaled) * valid_range
    return actions


def build_critic(states, actions, trainable, reuse, scope):
    """
    Define a critic network that predicts the Q-value of a given state and a
    given action Q(states, actions). This is obtained by feeding the network
    with the concatenation of the two inputs.

    Args:
        states   : a tensorflow placeholder containing the state of the 
                    environment
        actions  : a tensorflow placeholder containing the best action
                    according to the actor network
        trainable: whether the network is to be trained (main network) or to
                    have frozen weights (target network)
        reuse    : whether to reuse the weights and biases of an older network
                    with the same scope name
        scope    : the name of the tensorflow scope
    """
    with tf.variable_scope(scope):

        layer = tf.concat([states, actions], axis=1)

        # Convolution layers
        if hasattr(Settings, 'CONV_LAYERS') and Settings.CONV_LAYERS:
            for i, layer_settings in enumerate(Settings.CONV_LAYERS):
                layer = tf.layers.conv2d(inputs=layer,
                                         activation=tf.nn.relu,
                                         trainable=trainable,
                                         reuse=reuse,
                                         name='conv_'+str(i),
                                         **layer_settings)

            layer = tf.layers.flatten(layer)

        # Fully connected layers
        for i, nb_neurons in enumerate(Settings.HIDDEN_CRITIC_LAYERS):
            layer = tf.layers.dense(layer, nb_neurons,
                                    trainable=trainable,
                                    reuse=reuse,
                                    activation=tf.nn.relu,
                                    name='dense_'+str(i))

        q_values = tf.layers.dense(layer, 1,
                                   trainable=trainable, reuse=reuse,
                                   name='dense_last')
    return q_values


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
        tau : the update copy rate 
        name : the name of the operation
    """
    update_dest = []
    for src_var, dest_var in zip(src_vars, dest_vars):

        # Check if src_var and dest_var represents the same weight or bias
        src_name, dest_name = src_var.name, dest_var.name
        assert src_name[src_name.find("/"):] == dest_name[dest_name.find("/"):]

        op = dest_var.assign(tau * src_var + (1 - tau) * dest_var)
        update_dest.append(op)
    return tf.group(*update_dest, name=name)



def l2_regularization(vars):
    """
    Given a list of variables, return the sum of the squares of every variable
    corresponding to a weight (and not a biax).

    This is used to prevent the weights of the network to grow too much.
    """
    reg = 0
    for var in vars:
        if not 'bias' in var.name:
            reg += 1e-6 * tf.nn.l2_loss(var)
    return reg
