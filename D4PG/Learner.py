
import tensorflow as tf
import numpy as np

import Actor
from Model import *
from ExperienceBuffer import BUFFER

import settings


class Learner:

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
        self.is_not_done_ph = tf.placeholder(dtype=tf.float32,shape=[None], name='is_not_done_ph')

        # Main actor network
        self.actions = build_actor(self.state_ph, self.bounds, self.action_size,
                                   trainable=True, scope='learner_actor')

        # Main critic network
        self.q_distrib_of_given_actions = build_critic(
            self.state_ph, self.action_ph, trainable=True, reuse=False, scope='learner_critic')
        self.q_distrib_of_suggested_actions = build_critic(
            self.state_ph, self.actions, trainable=True, reuse=True, scope='learner_critic')
        
        min_q, max_q = settings.MIN_VALUE, settings.MAX_VALUE
        delta_z = (max_q - min_q) / (settings.NB_ATOMS - 1)
        self.z = [min_q + i * delta_z for i in range(settings.NB_ATOMS)]
        self.q_values_of_suggested_actions = tf.reduce_sum(self.z * self.q_distrib_of_suggested_actions, axis=1)

        # Target actor network
        self.target_next_actions = tf.stop_gradient(
            build_actor(self.next_state_ph, self.bounds, self.action_size,
                        trainable=False, scope='learner_target_actor'))

        # Target critic network
        self.q_distrib_next = tf.stop_gradient(
            build_critic(self.next_state_ph, self.target_next_actions,
                         trainable=False, reuse=False, scope='learner_target_critic'))

        # Isolate vars for each network
        self.actor_vars = get_vars('learner_actor', trainable=True)
        self.critic_vars = get_vars('learner_critic', trainable=True)
        self.vars = self.actor_vars + self.critic_vars

        self.target_actor_vars = get_vars('learner_target_actor', trainable=False)
        self.target_critic_vars = get_vars('learner_target_critic', trainable=False)
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

        # Compute the target value
        reward = tf.expand_dims(self.reward_ph, 1)
        not_done = tf.expand_dims(self.is_not_done_ph, 1)
        targets = reward + not_done * settings.DISCOUNT * self.q_distrib_next

        batch_size = tf.shape(self.reward_ph)[0]

        self.tour = []

        m = tf.zeros([batch_size, settings.NB_ATOMS])
        for i in range(settings.NB_ATOMS):
            Tz = tf.clip_by_value(reward + settings.DISCOUNT * self.z[i],
                                  min_q,
                                  max_q)
            bi = (Tz - min_q) / delta_z
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
                shape = (batch_size, settings.NB_ATOMS)
                scatter = tf.scatter_nd(indexes, values, shape)
                return (j + 1, m + scatter)

            _, m = tf.while_loop(cond, body, [j, m])

            self.tour.append([self.q_distrib_next[:, i], Tz, bi, l, u, l_index, u_index, m])

        critic_loss = -tf.reduce_sum(m * tf.log(self.q_distrib_of_given_actions))

        # Critic loss and optimization
        critic_loss += l2_regularization(self.critic_vars)
        critic_trainer = tf.train.AdamOptimizer(settings.CRITIC_LEARNING_RATE)
        self.critic_train_op = critic_trainer.minimize(critic_loss)

        # Actor loss and optimization
        actor_loss = -1 * tf.reduce_mean(self.q_values_of_suggested_actions)
        actor_loss += l2_regularization(self.actor_vars)
        actor_trainer = tf.train.AdamOptimizer(settings.ACTOR_LEARNING_RATE)
        self.actor_train_op = actor_trainer.minimize(actor_loss,
                                                     var_list=self.actor_vars)

        update_actors = []
        for i in range(settings.NB_ACTORS):
            op = copy_vars(self.actor_vars,
                           get_vars('worker_actor_%i'%(i+1), False),
                           1, 'update_actor_%i'%i)
            update_actors.append(op)
        self.update_actors = tf.group(*update_actors, name='update_actors')

    def run(self):

        total_eps = 1

        with self.sess.as_default(), self.sess.graph.as_default():

            self.sess.run(self.target_init)
            self.sess.run(self.update_actors)

            while not Actor.STOP_REQUESTED:
                
                batch = BUFFER.sample()

                if batch == []:
                    continue

                batch = batch[:8]

                feed_dict = {
                    self.state_ph: np.asarray([elem[0] for elem in batch]),
                    self.action_ph: np.asarray([elem[1] for elem in batch]),
                    self.reward_ph: np.asarray([elem[2] for elem in batch]),
                    self.next_state_ph: np.asarray([elem[3] for elem in batch]),
                    self.is_not_done_ph: np.asarray([elem[4] for elem in batch])
                }

                tour, _, _ = self.sess.run([self.tour, self.critic_train_op, self.actor_train_op],
                                     feed_dict=feed_dict)

                for i in range(settings.NB_ATOMS):
                    print("-"*100)
                    print("Distrib : ", tour[i][0], "\n")
                    print("Tz : ", tour[i][1], "\n")
                    print("bi : ", tour[i][2], "\n")
                    print("l : ", tour[i][3], "\n")
                    print("u : ", tour[i][4], "\n")
                    print("l_index : ", tour[i][5], "\n")
                    print("u_index : ", tour[i][6], "\n")
                    print("m : ", tour[i][7], "\n")
                1/0

                if total_eps % settings.UPDATE_TARGET_FREQ == 0:
                    # print("Orig : ", *self.sess.run(self.vars), sep='\n')
                    # print("Before : ", *self.sess.run(self.target_vars), sep='\n')
                    self.sess.run(self.update_targets)
                    # print("After : ", *self.sess.run(self.target_vars), sep='\n')
                    # 1/0

                if total_eps % settings.UPDATE_ACTORS_FREQ == 0:
                    self.sess.run(self.update_actors)

                total_eps += 1
