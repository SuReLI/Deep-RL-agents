
import numpy as np
import tensorflow as tf

from Model import build_model, copy_vars, get_vars
from settings import Settings


class QNetwork:

    def __init__(self, sess):
        print("Creation of the QNetwork...")

        self.sess = sess

        self.learning_rate = Settings.LEARNING_RATE

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

        self.build_model()
        self.build_target()
        self.get_variables()
        self.build_update_functions()
        self.build_train_operation()

        print("QNetwork created !")

    def build_model(self):

        #Q(s_t, .)
        self.Q_distrib = build_model(self.state_ph, True, False, 'main_network')

        # Q(s_t, a_t)
        ind = tf.stack((tf.range(Settings.BATCH_SIZE), self.action_ph), axis=1)
        self.Q_distrib_main_action = tf.gather_nd(self.Q_distrib, ind)

        # Q(s_{t+1}, .)
        self.Q_distrib_next = build_model(self.next_state_ph, True, True, 'main_network')
        self.Q_value_next = tf.reduce_sum(self.z * self.Q_distrib_next, axis=2)

        # argmax_a Q(s_{t+1}, a)
        self.best_next_action = tf.argmax(self.Q_value_next, 1, output_type=tf.int32)

    def build_target(self):

        # Q_target(s_{t+1}, .)
        self.Q_distrib_next_target = build_model(self.next_state_ph, False, False, 'target_network')

        # Q_target(s_{t+1}, argmax_a Q(s_{t+1}, a))
        ind = tf.stack((tf.range(Settings.BATCH_SIZE), self.best_next_action), axis=1)
        self.Q_distrib_next_target_best_action = tf.gather_nd(self.Q_distrib_next_target, ind)

    def get_variables(self):

        self.main_vars = get_vars('main_network', trainable=True)
        self.target_vars = get_vars('target_network', trainable=False)

    def build_update_functions(self):

        self.init_target_op = copy_vars(self.main_vars, self.target_vars,
                                        1, 'init_target')

        self.update_target_op = copy_vars(self.main_vars, self.target_vars,
                                          Settings.UPDATE_TARGET_RATE,
                                          'update_target')

    def build_train_operation(self):

        zz = tf.tile(self.z[None], [self.batch_size, 1])
        Tz = tf.clip_by_value(self.reward + Settings.DISCOUNT_N * self.not_done * zz,
                              Settings.MIN_Q, Settings.MAX_Q - 1e-4)
        bj = (Tz - Settings.MIN_Q) / self.delta_z
        l = tf.floor(bj)
        u = l + 1
        l_ind, u_ind = tf.to_int32(l), tf.to_int32(u)

        self.loss = tf.zeros([Settings.BATCH_SIZE])

        for j in range(Settings.NB_ATOMS):
            l_index = tf.stack((tf.range(self.batch_size), l_ind[:, j]), axis=1)
            u_index = tf.stack((tf.range(self.batch_size), u_ind[:, j]), axis=1)

            Q_distrib_l = tf.gather_nd(self.Q_distrib_main_action, l_index)
            Q_distrib_u = tf.gather_nd(self.Q_distrib_main_action, u_index)

            Q_distrib_l = tf.clip_by_value(Q_distrib_l, 1e-10, 1.0)
            Q_distrib_u = tf.clip_by_value(Q_distrib_u, 1e-10, 1.0)

            self.loss += self.Q_distrib_next_target_best_action[:, j] * (
                (u[:, j] - bj[:, j]) * tf.log(Q_distrib_l) +
                (bj[:, j] - l[:, j]) * tf.log(Q_distrib_u))

        self.weights = tf.placeholder(tf.float32, [None], name='weights')

        self.loss = tf.negative(self.loss)
        mean_loss = tf.reduce_mean(self.loss * self.weights)

        trainer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = trainer.minimize(mean_loss)

    def init_target(self):
        self.sess.run(self.init_target_op)

    def update_target(self):
        self.sess.run(self.update_target_op)

    def act(self, state):
        return self.sess.run(self.Q_distrib, feed_dict={self.state_ph: [state]})[0]

    def train(self, batch):

        feed_dict = {self.state_ph: batch[0],
                     self.action_ph: batch[1],
                     self.reward_ph: batch[2],
                     self.next_state_ph: batch[3],
                     self.not_done_ph: batch[4],
                     self.weights: batch[5]}
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)

        return loss
