
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
        self.q_values_of_given_actions = build_critic(
            self.state_ph, self.action_ph, trainable=True, reuse=False, scope='learner_critic')
        self.q_values_of_suggested_actions = build_critic(
            self.state_ph, self.actions, trainable=True, reuse=True, scope='learner_critic')


        # Target actor network
        self.target_next_actions = tf.stop_gradient(
            build_actor(self.next_state_ph, self.bounds, self.action_size,
                        trainable=False, scope='learner_target_actor'))

        # Target critic network
        self.q_values_next = tf.stop_gradient(
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
        self.target_init = copy_vars(self.critic_vars,
                                     self.target_critic_vars,
                                     1, 'init_critic_target')

        # Update values for target vars towards current actor and critic vars
        self.update_targets = copy_vars(self.vars,
                                        self.target_vars,
                                        settings.UPDATE_TARGET_RATE,
                                        'update_targets')

        # Compute the target value
        targets = tf.expand_dims(self.reward_ph, 1) + \
            tf.expand_dims(self.is_not_done_ph, 1) * settings.DISCOUNT * \
            self.q_values_next

        # 1-step temporal difference errors
        td_errors = targets - self.q_values_of_given_actions

        # Critic loss and optimization
        critic_loss = tf.reduce_mean(tf.square(td_errors))
        critic_trainer = tf.train.AdamOptimizer(settings.CRITIC_LEARNING_RATE)
        self.critic_train_op = critic_trainer.minimize(critic_loss)

        # Actor loss and optimization
        actor_loss = -1 * tf.reduce_mean(self.q_values_of_suggested_actions)
        actor_trainer = tf.train.AdamOptimizer(settings.ACTOR_LEARNING_RATE)
        self.actor_train_op = actor_trainer.minimize(actor_loss,
                                                     var_list=self.actor_vars)

        update_actors = []
        for i in range(settings.NB_ACTORS):
            zz = get_vars('worker_actor_%i'%(i+1), False)
            print(zz)
            op = copy_vars(self.actor_vars,
                           zz,
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

                feed_dict = {
                    self.state_ph: np.asarray([elem[0] for elem in batch]),
                    self.action_ph: np.asarray([elem[1] for elem in batch]),
                    self.reward_ph: np.asarray([elem[2] for elem in batch]),
                    self.next_state_ph: np.asarray([elem[3] for elem in batch]),
                    self.is_not_done_ph: np.asarray([elem[4] for elem in batch])
                }

                _, _ = self.sess.run([self.critic_train_op, self.actor_train_op],
                                     feed_dict=feed_dict)

                if total_eps % settings.UPDATE_TARGET_FREQ == 0:
                    # print("Orig : ", *self.sess.run(self.vars), sep='\n')
                    # print("Before : ", *self.sess.run(self.target_vars), sep='\n')
                    self.sess.run(self.update_targets)
                    # print("After : ", *self.sess.run(self.target_vars), sep='\n')
                    # 1/0

                if total_eps % settings.UPDATE_ACTORS_FREQ == 0:
                    self.sess.run(self.update_actors)

                total_eps += 1
