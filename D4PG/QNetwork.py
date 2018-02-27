
import tensorflow as tf
import numpy as np
import time

from Model import build_actor, build_critic
from network_utils import copy_vars, get_vars, l2_regularization

from settings import Settings


TOTAL_EPS = 0

class QNetwork:

    def __init__(self, sess, gui, saver, buffer):
        """
        Creation of the main and target networks and of the tensorflow
        operations to apply a gradient descent and update the target network.

        Args:
            sess  : the main tensorflow session in which to create the networks
            saver : a Saver instance to save the network weights
            buffer: the buffer that keeps the experiences to learn from
        """
        print("QNetwork initializing...")

        self.sess = sess
        self.gui = gui
        self.saver = saver
        self.buffer = buffer

        # Batch placeholders
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, *Settings.STATE_SIZE], name='state')
        self.action_ph = tf.placeholder(dtype=tf.float32, shape=[None, Settings.ACTION_SIZE], name='action')
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='reward')
        self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, *Settings.STATE_SIZE], name='next_state')
        self.not_done_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='not_done')
        
        # Turn these in column vector
        self.reward = tf.expand_dims(self.reward_ph, 1)
        self.not_done = tf.expand_dims(self.not_done_ph, 1)
        self.batch_size = tf.shape(self.reward_ph)[0]

        # Support of the distribution
        self.delta_z = (Settings.MAX_Q - Settings.MIN_Q) / (Settings.NB_ATOMS - 1)
        self.z = tf.range(Settings.MIN_Q, Settings.MAX_Q + self.delta_z, self.delta_z)

        # Build the networks
        self.build_model()
        self.build_target()
        self.build_update()
        self.build_train_operation()

        print("QNetwork created !\n")

    def build_model(self):
        """
        Build the main networks.
        
        To improve the critic network, we want to compute the cross-entropy loss
        between the projection on the support z of the target
        y = r_t + gamma * Q_target( s_{t+1}, A(s_{t+1}) ) and the Q-value at the
        time t Q(s_t, a_t) (with A(.) the output of the actor network).

        To improve the actor network, we apply the policy gradient :
        Grad = grad( Q(s_t, A(s_t)) ) * grad( A(s_t) )
        """

        # Compute A(s_t)
        self.actions = build_actor(self.state_ph, trainable=True, scope='learner_actor')

        # Compute Q(s_t, a_t)
        self.Q_distrib_given_actions = build_critic(self.state_ph, self.action_ph,
                                                       trainable=True, reuse=False,
                                                       scope='learner_critic')

        # Compute Q(s_t, A(s_t)) with the same network
        self.Q_distrib_suggested_actions = build_critic(self.state_ph, self.actions,
                                                       trainable=True, reuse=True,
                                                       scope='learner_critic')
        
        # Turn the distribution into value Qval(s_t, A(s_t))
        self.Q_values_suggested_actions = tf.reduce_sum(self.z * self.Q_distrib_suggested_actions, axis=1)


    def build_target(self):
        """
        Build the target networks.
        """
        # Compute A(s_{t+1})
        self.target_next_actions = build_actor(self.next_state_ph,
                                               trainable=False,
                                               scope='learner_target_actor')

        # Compute Q_target( s_{t+1}, A(s_{t+1}) )
        self.Q_distrib_next = build_critic(self.next_state_ph, self.target_next_actions,
                                           trainable=False, reuse=False,
                                           scope='learner_target_critic')

    def build_update(self):
        """
        Select the network variables and build the operation to copy main
        weights and biases to the target network.
        """
        # Isolate vars for each network
        self.actor_vars = get_vars('learner_actor', trainable=True)
        self.critic_vars = get_vars('learner_critic', trainable=True)
        self.vars = self.actor_vars + self.critic_vars

        self.target_actor_vars = get_vars('learner_target_actor', trainable=False)
        self.target_critic_vars = get_vars('learner_target_critic', trainable=False)
        self.target_vars = self.target_actor_vars + self.target_critic_vars


        # Initial operation to start with target_net == main_net
        self.init_target_op = copy_vars(self.vars, self.target_vars,
                                        1, 'init_target')

        # Update values for target vars towards current actor and critic vars
        self.target_update = copy_vars(self.vars, self.target_vars,
                                        Settings.UPDATE_TARGET_RATE,
                                        'target_update')

    def build_train_operation(self):
        """
        Apply the categorical algorithm to compute the cross-entropy loss
        (Cf https://arxiv.org/pdf/1707.06887.pdf)

        Because we apply the algoritm on whole batches and not sole experience
        and due to the architecture of tensorflow, the approach is slightly
        different : we don't keep track of the distribute probability of the
        projection in a vector m, instead for each atom we directly compute the
        product with log Q(s_t, a_t).

        Also apply the Policy Gradient Theorem to the actor network to learn
        from the previsions of the critic network.
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

        # Initialize the critic loss
        critic_loss = tf.zeros([self.batch_size])

        for j in range(Settings.NB_ATOMS):
            # Select the value of Q(s_t, a_t) onto the atoms l and u and clip it
            l_index = tf.stack((tf.range(self.batch_size), l_ind[:, j]), axis=1)
            u_index = tf.stack((tf.range(self.batch_size), u_ind[:, j]), axis=1)

            main_Q_distrib_l = tf.gather_nd(self.Q_distrib_given_actions, l_index)
            main_Q_distrib_u = tf.gather_nd(self.Q_distrib_given_actions, u_index)

            main_Q_distrib_l = tf.clip_by_value(main_Q_distrib_l, 1e-10, 1.0)
            main_Q_distrib_u = tf.clip_by_value(main_Q_distrib_u, 1e-10, 1.0)

            # loss +=   Q(s_{t+1}, a*) * (u - bj) * log Q[l](s_t, a_t)
            #         + Q(s_{t+1}, a*) * (bj - l) * log Q[u](s_t, a_t)
            critic_loss += self.Q_distrib_next[:, j] * (
                (u[:, j] - bj[:, j]) * tf.log(main_Q_distrib_l) +
                (bj[:, j] - l[:, j]) * tf.log(main_Q_distrib_u))

        # Take the mean loss on the batch
        critic_loss = tf.negative(critic_loss)
        critic_loss = tf.reduce_mean(critic_loss)
        critic_loss += l2_regularization(self.critic_vars)

        # Gradient descent
        critic_trainer = tf.train.AdamOptimizer(Settings.CRITIC_LEARNING_RATE)
        self.critic_train_op = critic_trainer.minimize(critic_loss)

        # Actor loss and optimization
        self.action_grad = tf.gradients(self.Q_values_suggested_actions, self.actions)[0]
        self.actor_grad = tf.gradients(self.actions, self.actor_vars, -self.action_grad)
        actor_trainer = tf.train.AdamOptimizer(Settings.ACTOR_LEARNING_RATE)
        self.actor_train_op = actor_trainer.apply_gradients(zip(self.actor_grad, self.actor_vars))

    def run(self):
        """
        Compute continuously gradient descents by sampling batches from the
        experience buffer.
        """
        global TOTAL_EPS

        self.total_eps = 1
        start_time = time.time()

        with self.sess.as_default(), self.sess.graph.as_default():

            self.sess.run(self.init_target_op)

            while not self.gui.STOP:
                
                if not self.buffer.buffer:
                    continue

                batch = np.asarray(self.buffer.sample())

                feed_dict = {self.state_ph: np.stack(batch[:, 0]),
                             self.action_ph: np.stack(batch[:, 1]),
                             self.reward_ph: batch[:, 2],
                             self.next_state_ph: np.stack(batch[:, 3]),
                             self.not_done_ph: batch[:, 4]}

                self.sess.run([self.critic_train_op, self.actor_train_op],
                               feed_dict=feed_dict)

                if self.total_eps % Settings.UPDATE_TARGET_FREQ == 0:
                    self.sess.run(self.target_update)

                if self.gui.save.get(self.total_eps):
                    self.saver.save(self.total_eps)

                # print("Learning ep : ", self.total_eps)
                self.total_eps += 1
                TOTAL_EPS += 1

                if self.total_eps % Settings.PERF_FREQ == 0:
                    print("PERF : %i learning round in %fs" %
                          (Settings.PERF_FREQ, time.time() - start_time))
                    start_time = time.time()
