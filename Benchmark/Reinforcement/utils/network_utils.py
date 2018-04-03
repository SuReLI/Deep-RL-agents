
import tensorflow as tf


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
