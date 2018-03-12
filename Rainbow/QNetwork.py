
import numpy as np
import tensorflow as tf

from Model import build_critic
from network_utils import copy_vars, get_vars
from settings import Settings


class QNetwork:

    def __init__(self, sess):
        """
        Creation of the main and target networks and of the tensorflow
        operations to apply a gradient descent and update the target network.

        Args:
            sess: the main tensorflow session in which to create the networks
        """
        print("Creation of the QNetwork...")

        self.sess = sess

        self.learning_rate = Settings.LEARNING_RATE
        self.delta_lr = Settings.LEARNING_RATE / Settings.TRAINING_EPS

        # Batch placeholders
        self.state_ph = tf.placeholder(tf.float32, [None, *Settings.STATE_SIZE], name='state')
        self.action_ph = tf.placeholder(tf.int32, [None], name='action')
        self.reward_ph = tf.placeholder(tf.float32, [None], name='reward')
        self.next_state_ph = tf.placeholder(tf.float32, [None, *Settings.STATE_SIZE], name='next_state')
        self.not_done_ph = tf.placeholder(tf.float32, [None], name='not_done')
        
        # Turn these in column vector to add them to the distribution
        self.reward = tf.expand_dims(self.reward_ph, 1)
        self.not_done = tf.expand_dims(self.not_done_ph, 1)
        self.batch_size = tf.shape(self.reward_ph)[0]

        # Support of the distribution
        self.delta_z = (Settings.MAX_Q - Settings.MIN_Q) / (Settings.NB_ATOMS - 1)
        self.z = tf.range(Settings.MIN_Q, Settings.MAX_Q + self.delta_z, self.delta_z)

        # Build the networks
        self.build_networks()
        self.build_train_operation()
        self.build_update()

        print("QNetwork created !\n")

    def build_networks(self):
        """
        Build the main network that predicts the Q-value distribution of a
        given state.
        
        Also build the operation to compute Q(s_t, a_t) for the gradient
        descent.
        Reminder :
            if simple DQN:
                y_t = r_t + gamma * max_a Q_target(s_{t+n}, a)
                    = r_t + gamma * Q_target( s_{t+n}, argmax_a Q_target(s_{t+n}, a) )
            elif double DQN:
                y_t = r_t + gamma * Q_target( s_{t+n}, argmax_a Q(s_{t+n}, a) )
            TD-error = y_t - Q(s_t, a_t)
        """
        # Compute Q(s_t, .)
        self.Q_st = build_critic(self.state_ph, trainable=True,
                                 reuse=False, scope='main_network')

        # Compute Q(s_t, a_t)
        ind = tf.stack((tf.range(Settings.BATCH_SIZE), self.action_ph), axis=1)
        self.Q_st_at = tf.gather_nd(self.Q_st, ind)

        # Compute Q_target(s_{t+n}, .)
        Q_target_st_n = build_critic(self.next_state_ph, trainable=False,
                                          reuse=False, scope='target_network')

        # If not double DQN, choose the best next action of the target network
        # Elif double DQN, choose the best next action of the main network
        if not Settings.DOUBLE_DQN:
            # Reuse Q_target(s_{t+n}, .)
            Q_st_n_max_a = Q_target_st_n
        else:
            # Compute Q(s_{t+n}, .)
            Q_st_n_max_a = build_critic(self.next_state_ph, trainable=True,
                                        reuse=True, scope='main_network')

        # Transform the distribution into the value to get the argmax
        if Settings.DISTRIBUTIONAL:
            Q_st_n_max_a = tf.reduce_sum(self.z * Q_st_n_max_a, axis=2)

        # Compute argmax_a Q[_target](s_{t+n}, a)
        best_at_n = tf.argmax(Q_st_n_max_a, 1, output_type=tf.int32)

        # Compute Q_target(s_{t+n}, argmax_a Q[_target](s_{t+n}, a))
        ind = tf.stack((tf.range(Settings.BATCH_SIZE), best_at_n), axis=1)
        self.Q_target_st_n_at_n = tf.gather_nd(Q_target_st_n, ind)

    def build_classical_loss(self):
        """
        Build the classical DQN loss :
        loss = [ target - Q(s_t, a_t) ]**2
        """

        target =  self.reward_ph + Settings.DISCOUNT_N * self.not_done_ph * self.Q_target_st_n_at_n
        self.loss = tf.square(target - self.Q_st_at)

    def build_distributional_loss(self):
        """
        Apply the categorical algorithm to compute the cross-entropy loss
        (Cf https://arxiv.org/pdf/1707.06887.pdf)

        Because we apply the algoritm on whole batches and not sole experience
        and due to the architecture of tensorflow, the approach is slightly
        different : we don't keep track of the distribute probability of the
        projection in a vector m, instead for each atom we directly compute the
        product with log Q(s_t, a_t)
        """
        
        # Extend the support for the whole batch (i.e. with batch_size lines)
        zz = tf.tile(self.z[None], [self.batch_size, 1])

        # Compute the projection of Tz onto the support z
        Tz = tf.clip_by_value(self.reward + Settings.DISCOUNT_N * self.not_done * zz,
                              Settings.MIN_Q, Settings.MAX_Q - 1e-4)
        bj = (Tz - Settings.MIN_Q) / self.delta_z
        l = tf.floor(bj)
        u = l + 1
        l_ind, u_ind = tf.to_int32(l), tf.to_int32(u)

        # initialize the loss
        self.loss = tf.zeros([Settings.BATCH_SIZE])

        for j in range(Settings.NB_ATOMS):
            # Select the value of Q(s_t, a_t) onto the atoms l and u and clip it
            l_index = tf.stack((tf.range(self.batch_size), l_ind[:, j]), axis=1)
            u_index = tf.stack((tf.range(self.batch_size), u_ind[:, j]), axis=1)

            Q_distrib_l = tf.gather_nd(self.Q_st_at, l_index)
            Q_distrib_u = tf.gather_nd(self.Q_st_at, u_index)

            Q_distrib_l = tf.clip_by_value(Q_distrib_l, 1e-10, 1.0)
            Q_distrib_u = tf.clip_by_value(Q_distrib_u, 1e-10, 1.0)

            # loss +=   Q(s_{t+n}, a*) * (u - bj) * log Q[l](s_t, a_t)
            #         + Q(s_{t+n}, a*) * (bj - l) * log Q[u](s_t, a_t)
            self.loss += self.Q_target_st_n_at_n[:, j] * (
                (u[:, j] - bj[:, j]) * tf.log(Q_distrib_l) +
                (bj[:, j] - l[:, j]) * tf.log(Q_distrib_u))


        # Take the mean loss on the batch
        self.loss = tf.negative(self.loss)        

    def build_train_operation(self):

        if Settings.DISTRIBUTIONAL:
            self.build_distributional_loss()
        else:
            self.build_classical_loss()

        self.weights = tf.placeholder(tf.float32, [None], name='weights')
        mean_loss = tf.reduce_mean(self.loss * self.weights)

        if Settings.PRIORITIZED_ER:
            mean_loss = tf.reduce_mean(self.loss * self.weights)
        else:
            mean_loss = tf.reduce_mean(self.loss)

        # Gradient descent
        trainer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = trainer.minimize(mean_loss)

    def build_update(self):
        """
        Select the network variables and build the operation to copy main
        weights and biases to the target network.
        """
        self.main_vars = get_vars('main_network', trainable=True)
        self.target_vars = get_vars('target_network', trainable=False)

        # Initial operation to start with target_net == main_net
        self.init_target_op = copy_vars(self.main_vars, self.target_vars,
                                        1, 'init_target')

        self.target_update = copy_vars(self.main_vars, self.target_vars,
                                       Settings.UPDATE_TARGET_RATE,
                                       'update_target')

    def init_target(self):
        """
        Wrapper method to initialize the target weights.
        """
        self.sess.run(self.init_target_op)

    def update_target(self):
        """
        Wrapper method to copy the main weights and biases to the target
        network.
        """
        self.sess.run(self.target_update)

    def act(self, state):
        """
        Wrapper method to compute the Q-value distribution given a single state.
        """
        return self.sess.run(self.Q_st, feed_dict={self.state_ph: [state]})[0]

    def decrease_lr(self):
        """
        Method to decrease the network learning rate.
        """
        if self.learning_rate > self.delta_lr:
            self.learning_rate -= self.delta_lr

    def train(self, batch, weights=None):
        """
        Wrapper method to train the network given a minibatch of experiences.
        """
        feed_dict = {self.state_ph: np.stack(batch[:, 0]),
                     self.action_ph: batch[:, 1],
                     self.reward_ph: batch[:, 2],
                     self.next_state_ph: np.stack(batch[:, 3]),
                     self.not_done_ph: batch[:, 4]}

        if Settings.PRIORITIZED_ER:
            feed_dict[self.weights] = weights
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            return loss

        else:
            self.sess.run(self.train_op, feed_dict=feed_dict)
