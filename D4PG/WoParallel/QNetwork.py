
import tensorflow as tf
import numpy as np

from Model import *
import settings


class Network:

    def __init__(self, sess, state_size, action_size, bounds):

        self.sess = sess

        self.state_size = state_size
        self.action_size = action_size
        self.bounds = bounds

        # placeholders
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size])
        self.action_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size])
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size])
        self.is_not_done_ph = tf.placeholder(dtype=tf.float32, shape=[None])

        self.min_q, self.max_q = settings.MIN_VALUE, settings.MAX_VALUE
        self.delta_z = (self.max_q - self.min_q) / (settings.NB_ATOMS - 1)
        self.z = [self.min_q + i * self.delta_z for i in range(settings.NB_ATOMS)]

        self.build_model()
        self.build_critic_loss()

    def build_model(self):

        # Main actor network
        self.actions = build_actor(self.state_ph, self.bounds, self.action_size,
                                   trainable=True, scope='actor')

        # Main critic network
        self.q_distrib_of_given_actions = build_critic(
            self.state_ph, self.action_ph, trainable=True, reuse=False, scope='critic')
        self.q_distrib_of_suggested_actions = build_critic(
            self.state_ph, self.actions, trainable=True, reuse=True, scope='critic')

        self.q_values_of_suggested_actions = tf.reduce_sum(self.z * self.q_distrib_of_suggested_actions, axis=1)
        
        # Target actor network
        self.target_next_actions = tf.stop_gradient(
            build_actor(self.next_state_ph, self.bounds, self.action_size,
                        trainable=False, scope='target_actor'))

        # Target critic network
        self.q_distrib_next = tf.stop_gradient(
            build_critic(self.next_state_ph, self.target_next_actions,
                         trainable=False, reuse=False, scope='target_critic'))

        # Isolate vars for each network
        self.actor_vars = get_vars('actor', trainable=True)
        self.critic_vars = get_vars('critic', trainable=True)
        self.vars = self.actor_vars + self.critic_vars

        self.target_actor_vars = get_vars('target_actor', trainable=False)
        self.target_critic_vars = get_vars('target_critic', trainable=False)
        self.target_vars = self.target_actor_vars + self.target_critic_vars

        # Initialize target critic vars to critic vars
        self.target_init = copy_vars(self.vars,
                                     self.target_vars,
                                     1, 'init_target')

        # Update values for target vars towards current actor and critic vars
        self.update_targets = copy_vars(self.vars,
                                        self.target_vars,
                                        settings.UPDATE_TARGET_RATE,
                                        'update_targets')

        # Actor loss and optimization
        actor_loss = -1 * tf.reduce_mean(self.q_values_of_suggested_actions)
        actor_loss += l2_regularization(self.actor_vars)
        actor_trainer = tf.train.AdamOptimizer(settings.ACTOR_LEARNING_RATE)
        self.actor_train_op = actor_trainer.minimize(actor_loss,
                                                     var_list=self.actor_vars)

    def build_critic_loss(self):

        # Compute the target value
        reward = tf.expand_dims(self.reward_ph, 1)
        not_done = tf.expand_dims(self.is_not_done_ph, 1)
        targets = reward + not_done * settings.DISCOUNT * self.q_distrib_next

        batch_size = tf.shape(self.reward_ph)[0]
        shape = (batch_size, settings.NB_ATOMS)

        m = tf.zeros([batch_size, settings.NB_ATOMS])
        for i in range(settings.NB_ATOMS):
            Tz = tf.clip_by_value(reward + settings.DISCOUNT * self.z[i],
                                  self.min_q,
                                  self.max_q)
            bi = (Tz - self.min_q) / self.delta_z
            l, u = tf.floor(bi), tf.ceil(bi)
            l = tf.reshape(l, [-1])
            u = tf.reshape(u, [-1])
            bi = tf.reshape(bi, [-1])
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
                values = [self.q_distrib_next[j, i] * (u[j] - bi[j]),
                          self.q_distrib_next[j, i] * (bi[j] - l[j])]
                return (j + 1, m + tf.scatter_nd(indexes, values, shape))

            _, m = tf.while_loop(cond, body, [j, m])

            # Critic loss and optimization
            critic_loss = -tf.reduce_sum(m * tf.log(self.q_distrib_of_given_actions))
            critic_loss += l2_regularization(self.critic_vars)
            critic_trainer = tf.train.AdamOptimizer(settings.CRITIC_LEARNING_RATE)
            self.critic_train_op = critic_trainer.minimize(critic_loss)

    def train(self, batch):

        states = np.asarray([elem[0] for elem in batch])
        actions = np.asarray([elem[1] for elem in batch])
        rewards = np.asarray([elem[2] for elem in batch])
        next_states = np.asarray([elem[3] for elem in batch])
        is_not_done = np.asarray([elem[4] for elem in batch])


        with self.sess.as_default(), self.sess.graph.as_default():
            feed_dict = {self.state_ph: states, 
                         self.action_ph: actions,
                         self.reward_ph: rewards,
                         self.next_state_ph: next_states,
                         self.is_not_done_ph: is_not_done}
            
            _, _ = self.sess.run([self.critic_train_op, self.actor_train_op],
                                  feed_dict=feed_dict)

            _ = self.sess.run(self.update_targets)
