
import tensorflow as tf
import numpy as np

from Model import build_model, get_vars, copy_vars
from settings import Settings


class QNetwork:

    def __init__(self, sess):
        print("Creation of the QNetwork...")

        self.sess = sess

        self.learning_rate = Settings.LEARNING_RATE
        self.delta_lr = Settings.LEARNING_RATE / Settings.TRAINING_STEPS

        # Placeholders
        self.state_ph = tf.placeholder(tf.float32, [None, *Settings.STATE_SIZE], name='state')
        self.action_ph = tf.placeholder(tf.int32, [None], name='action')
        self.reward_ph = tf.placeholder(tf.float32, [None], name='reward')
        self.next_state_ph = tf.placeholder(tf.float32, [None, *Settings.STATE_SIZE], name='next_state')
        self.not_done_ph = tf.placeholder(tf.float32, [None], name='not_done')
        
        # Turn these in column vector to add them to the distribution z
        self.reward = self.reward_ph[:, None]
        self.not_done = self.not_done_ph[:, None]
        self.batch_size = tf.shape(self.reward_ph)[0]

        # Support of the distribution
        self.delta_z = (Settings.MAX_Q - Settings.MIN_Q) / (Settings.NB_ATOMS - 1)
        self.z = tf.range(Settings.MIN_Q, Settings.MAX_Q + self.delta_z, self.delta_z)

        # Build the networks
        self.build_model()
        self.build_target()
        self.build_train_operation()
        self.build_update()

        print("QNetwork created !")

    def build_model(self):
                
        # Computation of Q(s, a) and argmax to get the next action to perform
        self.Q_distrib = build_model(self.state_ph,
                                     trainable=True, scope='main_network')

        self.Q_value = tf.reduce_sum(self.z * self.Q_distrib, axis=2)
        self.action = tf.argmax(self.Q_value, 1, output_type=tf.int32)

        ind = tf.stack((tf.range(self.batch_size), self.action_ph), axis=1)
        self.Q_distrib_taken_action = tf.gather_nd(self.Q_distrib, ind)

    def build_target(self):
        
        # Computation of Q(s', a), argmax to get a* and extraction of Q(s', a*)
        self.target_Q_distrib = build_model(self.next_state_ph,
                                            trainable=False, scope='target_network')
        
        self.target_Q_value = tf.reduce_sum(self.z * self.target_Q_distrib, axis=2)
        self.target_action = tf.argmax(self.target_Q_value, 1, output_type=tf.int32)

        ind = tf.stack((tf.range(self.batch_size), self.target_action), axis=1)
        self.target_Q_distrib_optimal_action = tf.gather_nd(self.target_Q_distrib, ind)


    def build_train_operation(self):

        zz = tf.tile(self.z[None], [self.batch_size, 1])
        Tz = tf.clip_by_value(self.reward + Settings.DISCOUNT * self.not_done * zz,
                              Settings.MIN_Q, Settings.MAX_Q - 1e-4)
        bj = (Tz - Settings.MIN_Q) / self.delta_z
        l = tf.floor(bj)
        u = l + 1
        l_ind, u_ind = tf.to_int32(l), tf.to_int32(u)

        self.loss = tf.zeros([self.batch_size])

        for j in range(Settings.NB_ATOMS):
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

    def build_update(self):
        self.main_vars = get_vars('main_network', trainable=True)
        self.target_vars = get_vars('target_network', trainable=False)

        self.init_target_update = copy_vars(self.main_vars, self.target_vars,
                                            1, 'init_target_update')

        self.target_update = copy_vars(self.main_vars, self.target_vars,
                                       Settings.UPDATE_TARGET_RATE,
                                       'target_update')


    def init_update_target(self):
        self.sess.run(self.init_target_update)

    def update_target(self):
        self.sess.run(self.target_update)

    def decrease_lr(self):
        if self.learning_rate > self.delta_lr:
            self.learning_rate -= self.delta_lr

    def train_minibatch(self, batch):

        state = batch[:, 0]
        action = batch[:, 1]
        reward = batch[:, 2]
        next_state = batch[:, 3]
        not_done = batch[:, 4]

        feed_dict = {self.state_ph: np.stack(state),
                     self.action_ph: action,
                     self.reward_ph: reward,
                     self.next_state_ph: np.stack(next_state),
                     self.not_done_ph: not_done}
        self.sess.run(self.train, feed_dict=feed_dict)
