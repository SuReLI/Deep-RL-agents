import numpy as np
import gym
import tensorflow as tf
import json
import sys
import os
from os import path
import random
from collections import deque

import parameters
from Environment import Environment
from ExperienceBuffer import ExperienceBuffer

##########################################################################
# Algorithm

# Deep Deterministic Policy Gradient (DDPG)
# An off-policy actor-critic algorithm that uses additive exploration noise (e.g. an Ornstein-Uhlenbeck process) on top
# of a deterministic policy to generate experiences (s, a, r, s'). It uses minibatches of these experiences from replay
# memory to update the actor (policy) and critic (Q function) parameters.
# Neural networks are used for function approximation.
# Slowly-changing "target" networks are used to improve stability and encourage convergence.
# Parameter updates are made via Adam.
# Assumes continuous action spaces!

##########################################################################
# Setup

# scale of the exploration noise process (1.0 is the range of each action
# dimension)
initial_noise_scale = 0.1
# decay rate (per episode) of the scale of the exploration noise process
noise_decay = 0.99
# mu parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt +
# sigma*dWt
exploration_mu = 0.0
# theta parameter for the exploration noise process: dXt =
# theta*(mu-Xt)*dt + sigma*dWt
exploration_theta = 0.15
# sigma parameter for the exploration noise process: dXt = theta*(mu-Xt
# )*dt + sigma*dWt
exploration_sigma = 0.2

# game parameters
env = Environment()
state_dim = env.get_state_size()[0]
action_dim = env.get_action_size()
low_bound, high_bound = env.get_bounds()

# set seeds to 0
env.seed(0)
np.random.seed(0)

# used for O(1) popleft() operation
replay_memory = ExperienceBuffer()


##########################################################################
# Tensorflow

tf.reset_default_graph()


class Agent:

    def __init__(self):
        pass

    def run(self, sess):

        # initialize session
        self.sess = sess
        print("MLJFDMLKFJ")
        self.sess.run(tf.global_variables_initializer())

        # placeholders
        state_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        action_ph = tf.placeholder(dtype=tf.float32, shape=[None, action_dim])
        reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        # indicators (go into target computation)
        is_not_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        is_training_ph = tf.placeholder(dtype=tf.bool, shape=())  # for dropout

        # episode counter
        episodes = tf.Variable(0.0, trainable=False, name='episodes')
        episode_inc_op = episodes.assign_add(1)

        # will use this to initialize both the actor network its slowly-changing
        # target network with same structure


        def generate_actor_network(s, trainable, reuse):
            hidden = tf.layers.dense(s, 8, activation=tf.nn.relu,
                                     trainable=trainable, name='dense', reuse=reuse)
            hidden_2 = tf.layers.dense(
                hidden, 8, activation=tf.nn.relu, trainable=trainable, name='dense_1', reuse=reuse)
            hidden_3 = tf.layers.dense(
                hidden_2, 8, activation=tf.nn.relu, trainable=trainable, name='dense_2', reuse=reuse)
            actions_unscaled = tf.layers.dense(
                hidden_3, action_dim, trainable=trainable, name='dense_3', reuse=reuse)
            # bound the actions to the valid range
            actions = low_bound + \
                tf.nn.sigmoid(actions_unscaled) * (high_bound - low_bound)
            return actions

        # actor network
        with tf.variable_scope('actor'):
            # Policy's outputted action for each state_ph (for generating actions and
            # training the critic)
            actions = generate_actor_network(state_ph, trainable=True, reuse=False)

        # slow target actor network
        with tf.variable_scope('slow_target_actor', reuse=False):
            # Slow target policy's outputted action for each next_state_ph (for training the critic)
            # use stop_gradient to treat the output values as constant targets when
            # doing backprop
            slow_target_next_actions = tf.stop_gradient(
                generate_actor_network(next_state_ph, trainable=False, reuse=False))

        # will use this to initialize both the critic network its slowly-changing
        # target network with same structure


        def generate_critic_network(s, a, trainable, reuse):
            state_action = tf.concat([s, a], axis=1)
            hidden = tf.layers.dense(state_action, 8, activation=tf.nn.relu,
                                     trainable=trainable, name='dense', reuse=reuse)
            hidden_2 = tf.layers.dense(
                hidden, 8, activation=tf.nn.relu, trainable=trainable, name='dense_1', reuse=reuse)
            hidden_3 = tf.layers.dense(
                hidden_2, 8, activation=tf.nn.relu, trainable=trainable, name='dense_2', reuse=reuse)
            q_values = tf.layers.dense(
                hidden_3, 1, trainable=trainable, name='dense_3', reuse=reuse)
            return q_values

        with tf.variable_scope('critic') as scope:
            # Critic applied to state_ph and a given action (for training critic)
            q_values_of_given_actions = generate_critic_network(
                state_ph, action_ph, trainable=True, reuse=False)
            # Critic applied to state_ph and the current policy's outputted actions
            # for state_ph (for training actor via deterministic policy gradient)
            q_values_of_suggested_actions = generate_critic_network(
                state_ph, actions, trainable=True, reuse=True)

        # slow target critic network
        with tf.variable_scope('slow_target_critic', reuse=False):
            # Slow target critic applied to slow target actor's outputted actions for
            # next_state_ph (for training critic)
            slow_q_values_next = tf.stop_gradient(generate_critic_network(
                next_state_ph, slow_target_next_actions, trainable=False, reuse=False))

        # isolate vars for each network
        actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        slow_target_actor_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_actor')
        critic_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        slow_target_critic_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_critic')

        # update values for slowly-changing targets towards current actor and critic
        update_slow_target_ops = []
        for i, slow_target_actor_var in enumerate(slow_target_actor_vars):
            update_slow_target_actor_op = slow_target_actor_var.assign(
                parameters.UPDATE_TARGET_RATE * actor_vars[i] + (1 - parameters.UPDATE_TARGET_RATE) * slow_target_actor_var)
            update_slow_target_ops.append(update_slow_target_actor_op)

        for i, slow_target_var in enumerate(slow_target_critic_vars):
            update_slow_target_critic_op = slow_target_var.assign(
                parameters.UPDATE_TARGET_RATE * critic_vars[i] + (1 - parameters.UPDATE_TARGET_RATE) * slow_target_var)
            update_slow_target_ops.append(update_slow_target_critic_op)

        update_slow_targets_op = tf.group(
            *update_slow_target_ops, name='update_slow_targets')

        # One step TD targets y_i for (s,a) from experience replay
        # = r_i + parameters.DISCOUNT*Q_slow(s',mu_slow(s')) if s' is not terminal
        # = r_i if s' terminal
        targets = tf.expand_dims(reward_ph, 1) + tf.expand_dims(is_not_terminal_ph,
                                                                1) * parameters.DISCOUNT * slow_q_values_next

        # 1-step temporal difference errors
        td_errors = targets - q_values_of_given_actions

        # critic loss function (mean-square value error with regularization)
        critic_loss = tf.reduce_mean(tf.square(td_errors))
        for var in critic_vars:
            if not 'bias' in var.name:
                critic_loss += 1e-6 * 0.5 * tf.nn.l2_loss(var)

        # critic optimizer
        critic_train_op = tf.train.AdamOptimizer(
            parameters.CRITIC_LEARNING_RATE).minimize(critic_loss)

        # actor loss function (mean Q-values under current policy with regularization)
        actor_loss = -1 * tf.reduce_mean(q_values_of_suggested_actions)
        for var in actor_vars:
            if not 'bias' in var.name:
                actor_loss += 1e-6 * 0.5 * tf.nn.l2_loss(var)

        # actor optimizer
        # the gradient of the mean Q-values wrt actor params is the deterministic
        # policy gradient (keeping critic params fixed)
        actor_train_op = tf.train.AdamOptimizer(
            parameters.ACTOR_LEARNING_RATE).minimize(actor_loss, var_list=actor_vars)


        total_steps = 0
        for ep in range(parameters.TRAINING_STEPS):

            total_reward = 0
            steps_in_ep = 0

            # Initialize exploration noise process
            noise_process = np.zeros(action_dim)
            noise_scale = (initial_noise_scale * noise_decay**ep) * \
                (high_bound - low_bound)

            # Initial state
            observation = self.env.reset()
            self.env.set_render(ep % 10 == 0)

            for t in range(parameters.MAX_EPISODE_STEPS):

                # choose action based on deterministic policy
                action_for_state, = self.sess.run(actions,
                                             feed_dict={state_ph: observation[None], is_training_ph: False})

                # add temporally-correlated exploration noise to action (using an Ornstein-Uhlenbeck process)
                # print(action_for_state)
                noise_process = exploration_theta * \
                    (exploration_mu - noise_process) + \
                    exploration_sigma * np.random.randn(action_dim)
                # print(noise_scale*noise_process)
                action_for_state += noise_scale * noise_process

                # take step
                next_observation, reward, done, _info = self.env.act(action_for_state)
                total_reward += reward

                replay_memory.add((observation, action_for_state, reward, next_observation,
                               # is next_observation a terminal state?
                               # 0.0 if done and not env.env._past_limit() else 1.0))
                               0.0 if done else 1.0))

                # update network weights to fit a minibatch of experience
                if total_steps % parameters.TRAINING_FREQ == 0 and len(replay_memory) >= parameters.BATCH_SIZE:

                    # grab N (s,a,r,s') tuples from replay memory
                    minibatch = replay_memory.sample()

                    # update the critic and actor params using mean-square value error
                    # and deterministic policy gradient, respectively
                    _, _ = self.sess.run([critic_train_op, actor_train_op],
                                    feed_dict={
                                    state_ph: np.asarray([elem[0] for elem in minibatch]),
                                    action_ph: np.asarray([elem[1] for elem in minibatch]),
                                    reward_ph: np.asarray([elem[2] for elem in minibatch]),
                                    next_state_ph: np.asarray([elem[3] for elem in minibatch]),
                                    is_not_terminal_ph: np.asarray([elem[4] for elem in minibatch]),
                                    is_training_ph: True})

                    # update slow actor and critic targets towards current actor and
                    # critic
                    _ = self.sess.run(update_slow_targets_op)

                observation = next_observation
                total_steps += 1
                steps_in_ep += 1

                if done:
                    # Increment episode counter
                    _ = self.sess.run(episode_inc_op)
                    break

            print('Episode %2i, Reward: %7.3f, Steps: %i, Final noise scale: %7.3f' %
                  (ep, total_reward, steps_in_ep, noise_scale))

        # Finalize and upload results
        self.env.close()
