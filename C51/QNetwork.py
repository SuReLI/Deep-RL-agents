
import tensorflow as tf

from Model import build_model

import settings


MIN_Q = settings.MIN_VALUE
MAX_Q = settings.MAX_VALUE


class QNetwork:

    def __init__(self, sess, state_size, action_size, scope):

        self.sess = sess

        with tf.variable_scope(scope):

            # Convolution network
            self.state, self.Q_distrib = build_model(state_size, action_size)

            # Support of the distribution
            self.delta_z = (MAX_Q - MIN_Q) / (settings.NB_ATOMS - 1)
            self.z = [MIN_Q + i * self.delta_z for i in range(settings.NB_ATOMS)]

            # Expected Q_value for each action in the state self.state
            self.Q_values = tf.reduce_sum(self.z * self.Q_distrib, axis=2)

            # Optimal action according to this Q value
            self.actions = tf.argmax(self.Q_values, 1, output_type=tf.int32)

            self.rewards = tf.placeholder(tf.float32, [None], name='reward_ph')
            self.done = tf.placeholder(tf.int32, [None], name='is_done_ph')

            batch_size = tf.shape(self.rewards)[0]
            shape = [batch_size, settings.NB_ATOMS]
            m = tf.zeros(shape)
            for i in range(settings.NB_ATOMS):
                Tz = tf.clip_by_value(self.rewards + settings.DISCOUNT * self.z[i],
                                      MIN_Q,
                                      MAX_Q)
                bi = (Tz - MIN_Q) / self.delta_z
                l, u = tf.floor(bi), tf.ceil(bi)
                l, u, bi = map(lambda x:tf.reshape(x, [-1]), (l, u, bi))
                l_index, u_index = tf.to_int32(l), tf.to_int32(u)

                # While-loop in tensorflow : we iterate over each exp in the batch
                # Loop counter
                j = tf.constant(0)

                # End condition
                cond = lambda j, m: tf.less(j, batch_size)

                # Function to apply in the loop : here, computation of the
                # distributed probability and projection over the old support
                # (c.f. C51 Algorithm 1) in a scattered tensor
                def body(j, m):
                    indexes = [(j, l_index[j]), (j, u_index[j])]
                    values = [self.Q_distrib[j, self.actions[j], i] * (u[j] - bi[j]),
                              self.Q_distrib[j, self.actions[j], i] * (bi[j] - l[j])]
                    return (j + 1, m + tf.scatter_nd(indexes, values, shape))

                _, m = tf.while_loop(cond, body, [j, m])

            self.main_Q_distrib = tf.placeholder(tf.float32, [None, settings.NB_ATOMS])
            self.loss = - tf.reduce_sum(m * tf.log(self.main_Q_distrib))

            self.trainer = tf.train.AdamOptimizer(
                learning_rate=settings.LEARNING_RATE)
            self.train = self.trainer.minimize(self.loss)

    def train_minibatch(self, batch):

        state = None
        action = None
        reward = None
        next_state = None
        done = None

        batch_size = len(reward)

        feed_dict = {self.state: state}
        Q_distrib = self.sess.run(self.Q_distrib, feed_dict=feed_dict)

        main_Q_distrib = [0] * batch_size
        for i in range(batch_size):
            main_Q_distrib[i] = Q_distrib[action[i]]

        feed_dict = {self.state: next_state,
                     self.main_Q_distrib: main_Q_distrib}
        _ = self.sess.run(self.train, feed_dict=feed_dict)
