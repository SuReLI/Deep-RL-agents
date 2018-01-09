
import tensorflow as tf
import numpy as np

from Model import build_model

import settings

MIN_Q = settings.MIN_VALUE
MAX_Q = settings.MAX_VALUE


class QNetwork:

    def __init__(self, sess, state_size, action_size, scope):

        self.sess = sess

        with tf.variable_scope(scope):

            self.test = []

            # Convolution network
            self.state_ph = tf.placeholder(tf.float32, [None, *state_size],
                            name='state_ph')

            self.Q_distrib = build_model(self.state_ph, action_size)

            # Support of the distribution
            delta_z = (MAX_Q - MIN_Q) / (settings.NB_ATOMS - 1)
            z = [MIN_Q + i * delta_z for i in range(settings.NB_ATOMS)]

            # Expected Q_value for each action in the state self.state
            self.Q_values = tf.reduce_sum(z * self.Q_distrib, axis=2)

            # Optimal action according to this Q value
            self.actions = tf.argmax(self.Q_values, 1, output_type=tf.int32)

            self.reward = tf.placeholder(tf.float32, [None], name='reward_ph')
            self.not_done = tf.placeholder(tf.bool, [None], name='not_done_ph')

            shape = [settings.BATCH_SIZE, settings.NB_ATOMS]


            def if_terminal(i, m):
                Tz = tf.clip_by_value(self.reward[i], MIN_Q, MAX_Q)
                bi = (Tz - MIN_Q) / delta_z
                l, u = tf.floor(bi), tf.ceil(bi)
                l, u, bi = map(lambda x:tf.reshape(x, [-1]), (l, u, bi))
                l_index, u_index = tf.to_int32(l), tf.to_int32(u)
                indexes = [(i, l_index[0]), (i, u_index[0])]
                values = [(u - bi)[0], (bi - l)[0]]
                return (m + tf.scatter_nd(indexes, values, shape))

            def if_not_terminal(i, m):
                for j in range(settings.NB_ATOMS):
                    Tz = tf.clip_by_value(self.reward[i] + settings.DISCOUNT * z[j],
                                          MIN_Q, MAX_Q)
                    bj = (Tz - MIN_Q) / delta_z
                    l, u = tf.floor(bj), tf.ceil(bj)
                    l, u, bj = map(lambda x:tf.reshape(x, [-1]), (l, u, bj))
                    l_index, u_index = tf.to_int32(l), tf.to_int32(u)
                    indexes = [(i, l_index[0]), (i, u_index[0])]
                    values = [self.Q_distrib[i, self.actions[i], j] * (u - bj),
                              self.Q_distrib[i, self.actions[i], j] * (bj - l)]
                    values = list(map(lambda x:x[0], values))
                    m = m + tf.scatter_nd(indexes, values, shape)
                return m

            m = tf.zeros(shape)
            for i in range(settings.BATCH_SIZE):
                m = tf.cond(self.not_done[i], lambda: if_not_terminal(i, m), lambda: if_terminal(i, m))
                print(i)

            self.main_Q_distrib = tf.placeholder(tf.float32, [None, settings.NB_ATOMS])
            self.loss = - tf.reduce_sum(m * tf.log(self.main_Q_distrib))

            self.trainer = tf.train.AdamOptimizer(
                learning_rate=settings.LEARNING_RATE)
            self.train = self.trainer.minimize(self.loss)

    def train_minibatch(self, batch):

        state = batch[:, 0]
        action = batch[:, 1]
        reward = batch[:, 2]
        next_state = batch[:, 3]
        not_done = batch[:, 4]

        if len(reward) != settings.BATCH_SIZE:
            return

        feed_dict = {self.state_ph: np.stack(state)}
        Q_distrib = self.sess.run(self.Q_distrib, feed_dict=feed_dict)

        main_Q_distrib = [0] * settings.BATCH_SIZE
        for i in range(settings.BATCH_SIZE):
            main_Q_distrib[i] = Q_distrib[i, action[i]]

        feed_dict = {self.state_ph: np.stack(next_state),
                     self.reward: reward,
                     self.not_done: not_done,
                     self.main_Q_distrib: main_Q_distrib}
        _ = self.sess.run(self.train, feed_dict=feed_dict)
