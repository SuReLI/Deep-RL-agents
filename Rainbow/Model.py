
import tensorflow as tf

from settings import Settings


def build_critic(states, trainable, reuse, scope):

    params = {'trainable':trainable, 'reuse':reuse}

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
            layer = tf.layers.dense(layer, nb_neurons,
                                    activation=tf.nn.relu,
                                    name='dense_'+str(i),
                                    **params)

        last_nb_neurons = Settings.HIDDEN_LAYERS[-1]

        if Settings.DUELING_DQN:

            adv_stream = tf.layers.dense(layer, last_nb_neurons,
                                         activation=tf.nn.relu,
                                         name='adv_stream', **params)

            value_stream = tf.layers.dense(layer, last_nb_neurons,
                                         activation=tf.nn.relu,
                                         name='value_stream', **params)

            if Settings.DISTRIBUTIONAL:
                # Advantage prediction
                advantage = tf.layers.dense(adv_stream, Settings.NB_ATOMS * Settings.ACTION_SIZE,
                                            name='adv', **params)
                advantage = tf.reshape(advantage, [-1, Settings.ACTION_SIZE, Settings.NB_ATOMS])
                advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)

                # Value prediction
                value = tf.layers.dense(value_stream, Settings.NB_ATOMS,
                                        name='value', **params)
                value = tf.reshape(value, [-1, 1, Settings.NB_ATOMS])
    
                # Qdistrib
                return tf.nn.softmax(value + advantage - advantage_mean, axis=2)

            else:
                advantage = tf.layers.dense(adv_stream, Settings.ACTION_SIZE,
                                            name='adv', **params)

                value = tf.layers.dense(value_stream, 1, name='value', **params)
                advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
                # Qvalues
                return value + tf.subtract(advantage, advantage_mean)

        else:
            layer = tf.layers.dense(layer, last_nb_neurons, activation=tf.nn.relu,
                                    name='last_dense', **params)

            if Settings.DISTRIBUTIONAL:
                # Distributional perspective : for each action, a fully-connected layer
                # with softmax activation predicts the Q-value distribution
                output = []
                for i in range(Settings.ACTION_SIZE):
                    output.append(tf.layers.dense(layer, Settings.NB_ATOMS,
                                                  activation=tf.nn.softmax,
                                                  name='output_' + str(i),
                                                  **params))
                # Qdistrib
                return tf.stack(output, axis=1)

            else:
                # Qvalues
                return tf.layers.dense(layer, Settings.ACTION_SIZE,
                                       name='output_layer', **params)
