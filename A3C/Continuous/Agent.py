
from time import time

import tensorflow as tf
import numpy as np
import scipy.signal
import random

from Environment import Environment
from MasterNetwork import Network
from Displayer import DISPLAYER

import parameters


# Discounting function used to calculate discounted returns.
def discount(x):
    return scipy.signal.lfilter([1], [1, -parameters.DISCOUNT], x[::-1],
                                axis=0)[::-1]


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope, tau):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


class Agent:

    epsilon = parameters.EPSILON_START
    epsilon_decay = (parameters.EPSILON_START - parameters.EPSILON_STOP) \
        / parameters.EPSILON_STEPS

    def __init__(self, worker_index, sess, render=False, master=False):
        print("Initialization of the agent", str(worker_index))

        self.worker_index = worker_index
        if master:
            self.name = 'global'
        else:
            self.name = 'Worker_' + str(worker_index)

        self.env = Environment()
        self.env.set_render(render)
        self.state_size = self.env.get_state_size()
        self.action_size = self.env.get_action_size()
        self.low_bound, self.high_bound = self.env.get_bounds()

        self.network = Network(sess, self.state_size, self.action_size,
                               self.low_bound, self.high_bound,
                               self.name)

        self.starting_time = 0

        if self.name != 'global':
            self.update_local_vars = update_target_graph(
                'global', self.name, 1)
            self.summary_writer = tf.summary.FileWriter("results/" + self.name,
                                                        sess.graph)

    def save(self):
        # Save model
        self.network.save_network(self.total_steps)

        # Save summary statistics
        summary = tf.Summary()
        summary.value.add(tag='Perf/Reward',
                          simple_value=np.mean(self.rewards_plus))
        summary.value.add(tag='Perf/Value',
                          simple_value=np.mean(self.next_values))
        summary.value.add(tag='Losses/Value',
                          simple_value=self.value_loss)
        summary.value.add(tag='Losses/Policy',
                          simple_value=self.policy_loss)
        summary.value.add(tag='Losses/Entropy',
                          simple_value=self.entropy)
        summary.value.add(tag='Losses/Grad Norm',
                          simple_value=self.grad_norm)
        self.summary_writer.add_summary(summary, self.total_steps)
        self.summary_writer.flush()

    def epsilon_decay(self):

        if Agent.epsilon > parameters.EPSILON_STOP:
            Agent.epsilon -= parameters.EPSILON_DECAY

    def work(self, sess, coord):
        print("Running", self.name, end='\n\n')
        self.starting_time = time()
        self.total_steps = 0
        ep = 0

        with sess.as_default(), sess.graph.as_default():
            with coord.stop_on_exception():

                while not coord.should_stop():

                    states_buffer = []
                    actions_buffer = []
                    rewards_buffer = []
                    next_state_buffer = []
                    done_buffer = []
                    episode_reward = 0
                    episode_step = 0

                    # Reset the local network to the global
                    sess.run(self.update_local_vars)

                    # Initialize the episode
                    s = self.env.reset()
                    done = False
                    max_steps = (parameters.MAX_EPISODE_STEP +
                                 ep // parameters.EP_ELONGATION)

                    while (not coord.should_stop() and
                           not done and episode_step < max_steps):

                        if random.random() < Agent.epsilon:
                            a = np.random.uniform(self.low_bound,
                                                  self.high_bound,
                                                  self.action_size)

                        else:
                            a = self.network.get_action([s])[0]

                        s_, r, done, _ = self.env.act(a)

                        # Store the experience
                        states_buffer.append(s)
                        actions_buffer.append(a)
                        rewards_buffer.append(r)
                        next_state_buffer.append(s_)
                        done_buffer.append(done)

                        if (len(states_buffer) >= parameters.MAX_LEN_BUFFER or
                                (len(states_buffer) != 0 and done)):

                            self.network.train(states_buffer,
                                               actions_buffer,
                                               discount(rewards_buffer),
                                               next_state_buffer,
                                               done_buffer)

                            states_buffer = []
                            actions_buffer = []
                            rewards_buffer = []
                            next_state_buffer = []
                            done_buffer = []

                            sess.run(self.update_local_vars)

                        episode_reward += r
                        s = s_

                        episode_step += 1
                        self.total_steps += 1

                    self.epsilon_decay()

                    ep += 1

                    if not coord.should_stop():
                        DISPLAYER.add_reward(episode_reward, self.worker_index)
                    if (self.worker_index == 1 and
                            ep % parameters.DISP_EP_REWARD_FREQ == 0):
                        print('Episode %2i, Reward: %7.3f, Steps: %i, '
                              'Epsilon: %7.3f, Max step: %i' %
                              (ep, episode_reward, episode_step,
                               Agent.epsilon, max_steps))

                    if (self.worker_index == 1 and
                            ep % parameters.SAVE_FREQ == 0):
                        self.save(ep)

                    if time() - self.starting_time > parameters.LIMIT_RUN_TIME:
                        coord.request_stop()

                self.summary_writer.close()
                self.close()

    def test(self, sess, number_run):
        print("Test session for %i runs" % number_run)

        with sess.as_default(), sess.graph.as_default():
            try:
                if self.name != 'global':
                    sess.run(self.update_local_vars)

                for _ in range(number_run):

                    s = self.env.reset()
                    episode_reward = 0
                    done = False

                    while not done:
                        a = self.network.get_action([s])[0]
                        s, r, done, _ = self.env.act(a)

                        episode_reward += r

                    print("Episode reward :", episode_reward)

            except KeyboardInterrupt as e:
                pass

            finally:
                print("End of test session")
                self.close()

    def close(self):
        self.env.close()
