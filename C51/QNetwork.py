
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
            z = tf.range(MIN_Q, MAX_Q + delta_z, delta_z)

            # Expected Q_value for each action in the state self.state
            self.Q_value = tf.reduce_sum(z * self.Q_distrib, axis=2)

            # Optimal action according to this Q value
            self.action = tf.argmax(self.Q_value, 1, output_type=tf.int32)

            self.reward_ph = tf.placeholder(tf.float32, [None], name='reward_ph')
            self.not_done_ph = tf.placeholder(tf.float32, [None], name='not_done_ph')

            batch_size = tf.shape(self.reward_ph)[0]
            shape = [batch_size, settings.NB_ATOMS]



###############################################################################
#                                                                             #
#                                                                             #
#                          /!\ WARNING /!\                                    #
#                     If type(bj) == int, then l == u == b                    #
#                            NO PROJECTION !                                  #
#                                                                             #
#       Replace reward and not done placeholder size with [None, 1] ???       #
#                                                                             #
#                                                                             #
#                                                                             #
###############################################################################

            zz = tf.tile(z[None], [batch_size, 1])

            self.reward = self.reward_ph[:, None]
            self.not_done = self.not_done_ph[:, None]

            Tz = tf.clip_by_value(self.reward + settings.DISCOUNT * self.not_done * zz,
                                  MIN_Q, MAX_Q)
            bj = (Tz - MIN_Q) / delta_z
            l, u = tf.floor(bj), tf.ceil(bj)
            l_ind, u_ind = tf.to_int32(l), tf.to_int32(u)
            print("*"*100)
            print(self.reward)
            print(self.not_done)
            print(self.not_done * zz)
            print(self.reward + self.not_done * zz)
            print("*"*100)

            ind = tf.stack((tf.range(batch_size), self.action), axis=1)
            Q_distrib_optimal_action = tf.gather_nd(self.Q_distrib, ind)


            self.main_Q_distrib = tf.placeholder(tf.float32, [None, settings.NB_ATOMS])
            self.loss = tf.zeros([batch_size])

            for j in range(settings.NB_ATOMS):
                print(tf.shape(l[:, j]))
                l_index = tf.stack((tf.range(batch_size), l_ind[:, j]), axis=1)
                u_index = tf.stack((tf.range(batch_size), u_ind[:, j]), axis=1)

                main_Q_distrib_l = tf.gather_nd(self.main_Q_distrib, l_index)
                main_Q_distrib_u = tf.gather_nd(self.main_Q_distrib, u_index)

                self.loss += Q_distrib_optimal_action[:, j] * (
                    (u[:, j] - bj[:, j]) * tf.log(main_Q_distrib_l) +
                    (bj[:, j] - l[:, j]) * tf.log(main_Q_distrib_u))

            self.trainer = tf.train.AdamOptimizer(settings.LEARNING_RATE)
            self.train = self.trainer.minimize(self.loss)

    def train_minibatch(self, batch):

        state = batch[:, 0]
        action = batch[:, 1]
        reward = batch[:, 2]
        next_state = batch[:, 3]
        not_done = batch[:, 4]

        batch_size = len(reward)

        feed_dict = {self.state_ph: np.stack(state)}
        Q_distrib = self.sess.run(self.Q_distrib, feed_dict=feed_dict)

        main_Q_distrib = [0] * batch_size
        for i in range(batch_size):
            main_Q_distrib[i] = Q_distrib[i, action[i]]

        feed_dict = {self.state_ph: np.stack(next_state),
                     self.reward_ph: reward,
                     self.not_done_ph: not_done,
                     self.main_Q_distrib: main_Q_distrib}
        _ = self.sess.run(self.train, feed_dict=feed_dict)