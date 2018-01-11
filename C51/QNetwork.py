
import tensorflow as tf
import numpy as np

from Model import build_model, get_vars, copy_vars

import settings

MIN_Q = settings.MIN_VALUE
MAX_Q = settings.MAX_VALUE


class QNetwork:

    def __init__(self, sess, state_size, action_size, trainable, scope):

        self.sess = sess

        self.state_size = state_size
        self.action_size = action_size

        # Support of the distribution
        self.delta_z = (MAX_Q - MIN_Q) / (settings.NB_ATOMS - 1)
        self.z = tf.range(MIN_Q, MAX_Q + self.delta_z, self.delta_z)

        self.build_action_prediction(trainable, scope)
        self.vars = get_vars(scope, trainable)
        if 'target' not in scope:
            self.build_train_operation()


    def build_action_prediction(self, trainable, scope):
        
        # Placeholders
        self.state_ph = tf.placeholder(tf.float32, [None, *self.state_size], name='state')
        self.reward_ph = tf.placeholder(tf.float32, [None], name='reward')
        self.not_done_ph = tf.placeholder(tf.float32, [None], name='not_done')

        # Turn these in column vector to add them to the distribution z
        self.reward = self.reward_ph[:, None]
        self.not_done = self.not_done_ph[:, None]

        self.batch_size = tf.shape(self.reward_ph)[0]
        
        self.Q_distrib = build_model(self.state_ph, self.action_size,
                                     trainable, scope)

        # Expected Q_value for each action in the state self.state
        self.Q_value = tf.reduce_sum(self.z * self.Q_distrib, axis=2)
        # Optimal action according to this Q value
        self.action = tf.argmax(self.Q_value, 1, output_type=tf.int32)

    def build_train_operation(self):

        zz = tf.tile(self.z[None], [self.batch_size, 1])
        Tz = tf.clip_by_value(self.reward + settings.DISCOUNT * self.not_done * zz,
                              MIN_Q, MAX_Q-1e-5)
        bj = (Tz - MIN_Q) / self.delta_z
        l = tf.floor(bj)
        u = l+1
        l_ind, u_ind = tf.to_int32(l), tf.to_int32(u)

        ind = tf.stack((tf.range(self.batch_size), self.action), axis=1)
        Q_distrib_optimal_action = tf.gather_nd(self.Q_distrib, ind)

        self.main_Q_distrib = tf.placeholder(tf.float32, [None, settings.NB_ATOMS])
        self.loss = tf.zeros([self.batch_size])

        for j in range(settings.NB_ATOMS):
            l_index = tf.stack((tf.range(self.batch_size), l_ind[:, j]), axis=1)
            u_index = tf.stack((tf.range(self.batch_size), u_ind[:, j]), axis=1)

            main_Q_distrib_l = tf.gather_nd(self.main_Q_distrib, l_index)
            main_Q_distrib_u = tf.gather_nd(self.main_Q_distrib, u_index)

            self.loss += Q_distrib_optimal_action[:, j] * (
                (u[:, j] - bj[:, j]) * tf.log(main_Q_distrib_l) +
                (bj[:, j] - l[:, j]) * tf.log(main_Q_distrib_u))

        self.loss = tf.negative(self.loss)
        self.loss = tf.reduce_mean(self.loss)

        self.trainer = tf.train.AdamOptimizer(settings.LEARNING_RATE)
        self.train = self.trainer.minimize(self.loss)

    def build_target_update(self, target_vars):
        self.init_target_update = copy_vars(self.vars, target_vars,
                                            1, 'init_target_update')

        self.target_update = copy_vars(self.vars, target_vars,
                                       settings.UPDATE_TARGET_RATE,
                                       'target_update')

    def init_update_target(self):
        _ = self.sess.run(self.init_target_update)

    def update_target(self):
        _ = self.sess.run(self.target_update)

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
