
import tensorflow as tf
import numpy as np

from settings import Settings


def fully_connected(*args, **kwargs):
    """
    Layer factory.
    """
    if Settings.NOISY:
        return noisy_layer(*args, **kwargs)
    else:
        return tf.layers.dense(*args, **kwargs)

def noisy_layer(inputs, units, activation=tf.identity, trainable=True, name=None, reuse=None):
    """
    Implementation of NoisyNets : layer with gaussian noise on its weights and
    biases. We use the Factorised Gaussian noise here (cf. paper)
    """

    with tf.variable_scope('noisy_layer', reuse=reuse):

        def f(x):
            return tf.multiply(tf.sign(x), tf.sqrt(tf.abs(x)))

        # Input and output size
        p = inputs.get_shape().as_list()[1]
        q = units

        # Weight initializer
        init_lim = 1 / np.sqrt(p)
        mu_init = tf.random_uniform_initializer(minval=-init_lim, 
                                                maxval=init_lim)
        sigma_init = tf.constant_initializer(0.5*init_lim)

        # Factorised Gaussian noise generation
        f_epsilon_i = f(tf.random_normal([p, 1]))
        f_epsilon_j = f(tf.random_normal([1, q]))

        epsilon_w = f_epsilon_i * f_epsilon_j
        epsilon_b = tf.squeeze(f_epsilon_j)

        # Weight noise
        mu_w = tf.get_variable(name + '/mu_w', [p, q], initializer=mu_init,
                               trainable=trainable)
        sigma_w = tf.get_variable(name +'/sigma_w', [p, q], initializer=sigma_init,
                                  trainable=trainable)
        w = mu_w + sigma_w * epsilon_w

        # Bias noise
        mu_b = tf.get_variable(name + '/mu_b', [q], initializer=mu_init,
                                trainable=trainable)
        sigma_b = tf.get_variable(name +'/sigma_b', [q], initializer=sigma_init,
                                  trainable=trainable)
        b = mu_b + sigma_b * epsilon_b

        return activation(tf.matmul(inputs, w) + b)


def build_critic(states, trainable, reuse, scope):

    params = {'trainable': trainable, 'reuse': reuse}

    with tf.variable_scope(scope):

        layer = states
        
        # Convolution layers
        if hasattr(Settings, 'CONV_LAYERS') and Settings.CONV_LAYERS:
            for i, layer_settings in enumerate(Settings.CONV_LAYERS):
                layer = tf.layers.conv2d(inputs=layer,
                                         activation=tf.nn.relu,
                                         name='conv_'+str(i),
                                         **params,
                                         **layer_settings)

            layer = tf.layers.flatten(layer)

        # Fully connected layers
        for i, nb_neurons in enumerate(Settings.HIDDEN_LAYERS[:-1]):
            layer = fully_connected(layer, nb_neurons,
                                    activation=tf.nn.relu,
                                    name='dense_'+str(i),
                                    **params)

        last_nb_neurons = Settings.HIDDEN_LAYERS[-1]

        if Settings.DUELING_DQN:

            adv_stream = fully_connected(layer, last_nb_neurons,
                                         activation=tf.nn.relu,
                                         name='adv_stream', **params)

            value_stream = fully_connected(layer, last_nb_neurons,
                                           activation=tf.nn.relu,
                                           name='value_stream', **params)

            if Settings.DISTRIBUTIONAL:
                # Advantage prediction
                advantage = fully_connected(adv_stream, Settings.NB_ATOMS * Settings.ACTION_SIZE,
                                            name='adv', **params)
                advantage = tf.reshape(advantage, [-1, Settings.ACTION_SIZE, Settings.NB_ATOMS])
                advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)

                # Value prediction
                value = fully_connected(value_stream, Settings.NB_ATOMS,
                                        name='value', **params)
                value = tf.reshape(value, [-1, 1, Settings.NB_ATOMS])
    
                # Qdistrib
                return tf.nn.softmax(value + advantage - advantage_mean, axis=2)

            else:
                advantage = fully_connected(adv_stream, Settings.ACTION_SIZE,
                                            name='adv', **params)

                value = fully_connected(value_stream, 1, name='value', **params)
                advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
                # Qvalues
                return value + tf.subtract(advantage, advantage_mean)

        else:
            layer = fully_connected(layer, last_nb_neurons, activation=tf.nn.relu,
                                    name='last_dense', **params)

            if Settings.DISTRIBUTIONAL:
                # Distributional perspective : for each action, a fully-connected layer
                # with softmax activation predicts the Q-value distribution
                output = []
                for i in range(Settings.ACTION_SIZE):
                    output.append(fully_connected(layer, Settings.NB_ATOMS,
                                                  activation=tf.nn.softmax,
                                                  name='output_' + str(i),
                                                  **params))
                # Qdistrib
                return tf.stack(output, axis=1)

            else:
                # Qvalues
                return fully_connected(layer, Settings.ACTION_SIZE,
                                       name='output_layer', **params)
