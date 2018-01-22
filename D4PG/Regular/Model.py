
import tensorflow as tf
import settings


def build_actor(states, bounds, action_size, trainable, scope):
    with tf.variable_scope(scope):
        hidden = tf.layers.dense(states, 8, trainable=trainable,
                                 activation=tf.nn.relu, name='dense')
        hidden_2 = tf.layers.dense(hidden, 8, trainable=trainable,
                                   activation=tf.nn.relu, name='dense_1')
        hidden_3 = tf.layers.dense(hidden_2, 8, trainable=trainable,
                                   activation=tf.nn.relu, name='dense_2')
        actions_unscaled = tf.layers.dense(hidden_3, action_size,
                                           trainable=trainable, name='dense_3')
        # bound the actions to the valid range
        low_bound, high_bound = bounds
        valid_range = high_bound - low_bound
        actions = low_bound + tf.nn.sigmoid(actions_unscaled) * valid_range
    return actions


def build_critic(states, actions, trainable, reuse, scope):
    with tf.variable_scope(scope):
        states_actions = tf.concat([states, actions], axis=1)
        hidden = tf.layers.dense(states_actions, 8,
                                 trainable=trainable, reuse=reuse,
                                 activation=tf.nn.relu, name='dense')
        hidden_2 = tf.layers.dense(hidden, 8,
                                   trainable=trainable, reuse=reuse,
                                   activation=tf.nn.relu, name='dense_1')
        hidden_3 = tf.layers.dense(hidden_2, 8,
                                   trainable=trainable, reuse=reuse,
                                   activation=tf.nn.relu, name='dense_2')
        Q_values = tf.layers.dense(hidden_3, settings.NB_ATOMS,
                                   trainable=trainable, reuse=reuse,
                                   activation=tf.nn.softmax, name='dense_3')
    return Q_values


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

def l2_regularization(vars):
    reg = 0
    for var in vars:
        if not 'bias' in var.name:
            reg += 1e-6 * 0.5 * tf.nn.l2_loss(var)
    return reg
