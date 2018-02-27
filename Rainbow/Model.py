
import tensorflow as tf

from settings import Settings


def build_critic(states, trainable, reuse, scope):

    with tf.variable_scope(scope):

        layer = states
        
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
        for i, nb_neurons in enumerate(Settings.HIDDEN_LAYERS[:-1]):
            layer = tf.layers.dense(layer, nb_neurons,
                                    trainable=trainable, reuse=reuse,
                                    activation=tf.nn.relu,
                                    name='dense_'+str(i))

        last_nb_neurons = Settings.HIDDEN_LAYERS[-1]

        # Advantage prediction
        adv_stream = tf.layers.dense(layer, last_nb_neurons,
                                     trainable=trainable, reuse=reuse,
                                     activation=tf.nn.relu, name='adv_stream')

        advantage = tf.layers.dense(adv_stream, Settings.NB_ATOMS * Settings.ACTION_SIZE,
                                    trainable=trainable, reuse=reuse, name='adv')
        advantage = tf.reshape(advantage, [-1, Settings.ACTION_SIZE, Settings.NB_ATOMS])

        advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)

        # Value prediction
        value_stream = tf.layers.dense(layer, last_nb_neurons,
                                     trainable=trainable, reuse=reuse,
                                     activation=tf.nn.relu, name='value_stream')
        value = tf.layers.dense(value_stream, Settings.NB_ATOMS,
                                     trainable=trainable, reuse=reuse,
                                     activation=tf.nn.relu, name='value')
        value = tf.reshape(value, [-1, 1, Settings.NB_ATOMS])

        Qdistrib = tf.nn.softmax(value + advantage - advantage_mean, axis=2)

    return Qdistrib
