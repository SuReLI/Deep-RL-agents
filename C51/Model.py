
import tensorflow as tf
import settings

def build_model(state_size, action_size):

    inputs = tf.placeholder(tf.float32, [None, *state_size],
                            name='state_ph')

    if settings.CONV:
        with tf.variable_scope('Convolutional_Layers'):
            conv1 = tf.layers.conv2d(inputs=inputs,
                                     filters=32,
                                     kernel_size=[8, 8],
                                     strides=[4, 4],
                                     padding='valid',
                                     activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv1, 64, [4, 4], [2, 2], 'valid',
                                     activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(conv2, 64, [3, 3], [1, 1], 'valid',
                                     activation=tf.nn.relu)

        # Flatten the output
        hidden = tf.layers.flatten(conv3)

    else:
        hidden = tf.layers.dense(inputs, 64, tf.nn.relu, name='hidden1')
        hidden = tf.layers.dense(hidden, 64, tf.nn.relu, name='hidden2')

    output = []
    for i in range(action_size):
        output.append(tf.layers.dense(hidden, settings.NB_ATOMS,
                                      activation=tf.nn.softmax,
                                      name='hidden3_' + str(i+1)))
    output = tf.stack(output, axis=1)

    return inputs, output
