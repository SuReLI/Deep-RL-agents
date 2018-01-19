
import tensorflow as tf
import numpy as np

from Model import *
import settings

MIN_Q = settings.MIN_VALUE
MAX_Q = settings.MAX_VALUE


class Network:

    def __init__(self, sess, state_size, action_size, bounds):

        self.sess = sess

        self.state_size = state_size
        self.action_size = action_size
        self.bounds = bounds

        # placeholders
        self.state_ph = tf.placeholder(dtype=tf.float32,shape=[None, self.state_size], name='state_ph')
        self.action_ph = tf.placeholder(dtype=tf.float32,shape=[None, self.action_size], name='action_ph')
        self.reward_ph = tf.placeholder(dtype=tf.float32,shape=[None], name='reward_ph')
        self.next_state_ph = tf.placeholder(dtype=tf.float32,shape=[None, self.state_size], name='next_state_ph')
        self.not_done_ph = tf.placeholder(dtype=tf.float32,shape=[None], name='not_done_ph')

        # Turn these in column vector to add them to the distribution z
        self.reward = self.reward_ph[:, None]
        self.not_done = self.not_done_ph[:, None]
        self.batch_size = tf.shape(self.reward_ph)[0]

        # Support of the distribution
        self.delta_z = (MAX_Q - MIN_Q) / (settings.NB_ATOMS - 1)
        self.z = tf.range(MIN_Q, MAX_Q + self.delta_z, self.delta_z)

        self.build_model()
        self.build_target()
        self.get_variables()
        self.build_train_operation()
        self.build_update_functions()

    def build_model(self):

        # Main actor network
        self.actions = build_actor(self.state_ph, self.bounds, self.action_size,
                                   trainable=True, scope='learner_actor')

        # Main critic network
        self.Q_distrib_given_actions = build_critic(self.state_ph, self.action_ph,
                                                       trainable=True, reuse=False,
                                                       scope='learner_critic')
        self.Q_distrib_suggested_actions = build_critic(self.state_ph, self.actions,
                                                       trainable=True, reuse=True,
                                                       scope='learner_critic')
        
        self.Q_values_suggested_actions = tf.reduce_sum(self.z * self.Q_distrib_suggested_actions, axis=1)
        
    def build_target(self):

        # Target actor network
        self.target_next_actions = tf.stop_gradient(
            build_actor(self.next_state_ph, self.bounds, self.action_size,
                        trainable=False, scope='learner_target_actor'))

        # Target critic network
        self.Q_distrib_next = tf.stop_gradient(
            build_critic(self.next_state_ph, self.target_next_actions,
                         trainable=False, reuse=False, scope='learner_target_critic'))

    def get_variables(self):
        # Isolate vars for each network
        self.actor_vars = get_vars('learner_actor', trainable=True)
        self.critic_vars = get_vars('learner_critic', trainable=True)
        self.vars = self.actor_vars + self.critic_vars

        self.target_actor_vars = get_vars('learner_target_actor', trainable=False)
        self.target_critic_vars = get_vars('learner_target_critic', trainable=False)
        self.target_vars = self.target_actor_vars + self.target_critic_vars

    def build_update_functions(self):

        # Initialize target critic vars to critic vars
        self.target_init = copy_vars(self.vars,
                                     self.target_vars,
                                     1, 'init_target')

        # Update values for target vars towards current actor and critic vars
        self.update_targets = copy_vars(self.vars,
                                        self.target_vars,
                                        settings.UPDATE_TARGET_RATE,
                                        'update_targets')

    def build_train_operation(self):

        zz = tf.tile(self.z[None], [self.batch_size, 1])
        Tz = tf.clip_by_value(self.reward + settings.DISCOUNT * self.not_done * zz,
                              MIN_Q, MAX_Q - 1e-5)
        bj = (Tz - MIN_Q) / self.delta_z
        l = tf.floor(bj)
        u = l + 1
        l_ind, u_ind = tf.to_int32(l), tf.to_int32(u)

        critic_loss = tf.zeros([self.batch_size])

        for j in range(settings.NB_ATOMS):
            l_index = tf.stack((tf.range(self.batch_size), l_ind[:, j]), axis=1)
            u_index = tf.stack((tf.range(self.batch_size), u_ind[:, j]), axis=1)

            main_Q_distrib_l = tf.gather_nd(self.Q_distrib_given_actions, l_index)
            main_Q_distrib_u = tf.gather_nd(self.Q_distrib_given_actions, u_index)

            main_Q_distrib_l = tf.clip_by_value(main_Q_distrib_l, 1e-10, 1.0)
            main_Q_distrib_u = tf.clip_by_value(main_Q_distrib_u, 1e-10, 1.0)

            critic_loss += self.Q_distrib_next[:, j] * (
                (u[:, j] - bj[:, j]) * tf.log(main_Q_distrib_l) +
                (bj[:, j] - l[:, j]) * tf.log(main_Q_distrib_u))

        critic_loss = tf.negative(critic_loss)
        critic_loss = tf.reduce_mean(critic_loss)

        # Critic loss and optimization
        critic_loss += l2_regularization(self.critic_vars)
        critic_trainer = tf.train.AdamOptimizer(settings.CRITIC_LEARNING_RATE)
        self.critic_train_op = critic_trainer.minimize(critic_loss)

        # Actor loss and optimization
        self.action_grad = tf.gradients(self.Q_values_suggested_actions, self.actions)[0]
        self.actor_grad = tf.gradients(self.actions, self.actor_vars, -self.action_grad)
        actor_trainer = tf.train.AdamOptimizer(settings.ACTOR_LEARNING_RATE)
        self.actor_train_op = actor_trainer.apply_gradients(zip(self.actor_grad, self.actor_vars))

    def train(self, batch):

        states = np.asarray([elem[0] for elem in batch])
        actions = np.asarray([elem[1] for elem in batch])
        rewards = np.asarray([elem[2] for elem in batch])
        next_states = np.asarray([elem[3] for elem in batch])
        not_done = np.asarray([elem[4] for elem in batch])


        with self.sess.as_default(), self.sess.graph.as_default():
            feed_dict = {self.state_ph: states, 
                         self.action_ph: actions,
                         self.reward_ph: rewards,
                         self.next_state_ph: next_states,
                         self.not_done_ph: not_done}
            
            _, _ = self.sess.run([self.critic_train_op, self.actor_train_op],
                                  feed_dict=feed_dict)

            _ = self.sess.run(self.update_targets)
