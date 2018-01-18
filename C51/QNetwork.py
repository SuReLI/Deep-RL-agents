
import tensorflow as tf
import numpy as np

from Model import build_model, get_vars, copy_vars

import settings

MIN_Q = settings.MIN_VALUE
MAX_Q = settings.MAX_VALUE


class QNetwork:

    def __init__(self, sess, state_size, action_size):

        self.sess = sess

        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = settings.LEARNING_RATE
        self.delta_lr = settings.LEARNING_RATE / settings.TRAINING_STEPS

        # Placeholders
        self.state_ph = tf.placeholder(tf.float32, [None, *self.state_size], name='state')
        self.action_ph = tf.placeholder(tf.int32, [None], name='action')
        self.reward_ph = tf.placeholder(tf.float32, [None], name='reward')
        self.next_state_ph = tf.placeholder(tf.float32, [None, *self.state_size], name='next_state')
        self.not_done_ph = tf.placeholder(tf.float32, [None], name='not_done')
        
        # Turn these in column vector to add them to the distribution z
        self.reward = self.reward_ph[:, None]
        self.not_done = self.not_done_ph[:, None]
        self.batch_size = tf.shape(self.reward_ph)[0]

        # Support of the distribution
        self.delta_z = (MAX_Q - MIN_Q) / (settings.NB_ATOMS - 1)
        self.z = tf.range(MIN_Q, MAX_Q + self.delta_z, self.delta_z)

        # Build the networks
        self.build_model()
        self.build_target()
        self.build_train_operation()

        self.main_vars = get_vars('main_network', trainable=True)
        self.target_vars = get_vars('target_network', trainable=False)

        self.init_target_update = copy_vars(self.main_vars, self.target_vars,
                                            1, 'init_target_update')

        self.target_update = copy_vars(self.main_vars, self.target_vars,
                                       settings.UPDATE_TARGET_RATE,
                                       'target_update')

    def build_model(self):
                
        # Computation of Q(s, a) and argmax to get the next action to perform
        self.Q_distrib = build_model(self.state_ph, self.action_size,
                                     trainable=True, scope='main_network')

        self.Q_value = tf.reduce_sum(self.z * self.Q_distrib, axis=2)
        self.action = tf.argmax(self.Q_value, 1, output_type=tf.int32)

        ind = tf.stack((tf.range(self.batch_size), self.action_ph), axis=1)
        self.Q_distrib_taken_action = tf.gather_nd(self.Q_distrib, ind)

    def build_target(self):
        
        # Computation of Q(s', a), argmax to get a* and extraction of Q(s', a*)
        self.target_Q_distrib = build_model(self.next_state_ph, self.action_size,
                                            trainable=False, scope='target_network')
        
        self.target_Q_value = tf.reduce_sum(self.z * self.target_Q_distrib, axis=2)
        self.target_action = tf.argmax(self.target_Q_value, 1, output_type=tf.int32)

        ind = tf.stack((tf.range(self.batch_size), self.target_action), axis=1)
        self.target_Q_distrib_optimal_action = tf.gather_nd(self.target_Q_distrib, ind)


    def build_train_operation(self):

        zz = tf.tile(self.z[None], [self.batch_size, 1])
        Tz = tf.clip_by_value(self.reward + settings.DISCOUNT * self.not_done * zz,
                              MIN_Q, MAX_Q - 1e-4)
        bj = (Tz - MIN_Q) / self.delta_z
        l = tf.floor(bj)
        u = l + 1
        l_ind, u_ind = tf.to_int32(l), tf.to_int32(u)

        self.loss = tf.zeros([self.batch_size])

        for j in range(settings.NB_ATOMS):
            l_index = tf.stack((tf.range(self.batch_size), l_ind[:, j]), axis=1)
            u_index = tf.stack((tf.range(self.batch_size), u_ind[:, j]), axis=1)

            main_Q_distrib_l = tf.gather_nd(self.Q_distrib_taken_action, l_index)
            main_Q_distrib_u = tf.gather_nd(self.Q_distrib_taken_action, u_index)

            main_Q_distrib_l = tf.clip_by_value(main_Q_distrib_l, 1e-10, 1.0)
            main_Q_distrib_u = tf.clip_by_value(main_Q_distrib_u, 1e-10, 1.0)

            self.loss += self.target_Q_distrib_optimal_action[:, j] * (
                (u[:, j] - bj[:, j]) * tf.log(main_Q_distrib_l) +
                (bj[:, j] - l[:, j]) * tf.log(main_Q_distrib_u))

        self.loss = tf.negative(self.loss)
        self.loss = tf.reduce_mean(self.loss)

        self.trainer = tf.train.AdamOptimizer(self.learning_rate*1e-4)
        self.train = self.trainer.minimize(self.loss)

    def init_update_target(self):
        _ = self.sess.run(self.init_target_update)

    def update_target(self):
        _ = self.sess.run(self.target_update)

    def decrease_lr(self):
        if self.learning_rate > self.delta_lr:
            self.learning_rate -= self.delta_lr

    def train_minibatch(self, batch):

        state = batch[:, 0]
        action = batch[:, 1]
        reward = batch[:, 2]
        next_state = batch[:, 3]
        not_done = batch[:, 4]

        batch_size = len(reward)

        feed_dict = {self.state_ph: np.stack(state),
                     self.action_ph: action,
                     self.reward_ph: reward,
                     self.next_state_ph: np.stack(next_state),
                     self.not_done_ph: not_done}
        _ = self.sess.run(self.train, feed_dict=feed_dict)
