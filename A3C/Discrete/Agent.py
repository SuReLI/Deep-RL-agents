
from time import time

import tensorflow as tf
import numpy as np
import scipy.signal
import random

from Environment import Environment
from MasterNetwork import Network
from Displayer import DISPLAYER
from Saver import SAVER

import parameters


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1],
                                [1, -gamma],
                                x[::-1],
                                axis=0)[::-1]


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


class Agent:

    def __init__(self, worker_index, sess, render=False, master=False):

        self.worker_index = worker_index
        if master:
            self.name = 'global'
        else:
            print("Initialization of the agent", str(worker_index))
            self.name = 'Worker_' + str(worker_index)

        self.env = Environment()
        self.env.set_render(render)
        self.state_size = self.env.get_state_size()
        self.action_size = self.env.get_action_size()

        self.network = Network(self.state_size, self.action_size, self.name)
        self.update_local_vars = update_target_graph('global', self.name)

        self.starting_time = 0
        self.epsilon = parameters.EPSILON_START

        if self.name != 'global':
            self.summary_writer = tf.summary.FileWriter("results/" + self.name,
                                                        sess.graph)

    def save(self, episode_step):
        # Save model
        SAVER.save(episode_step)

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
        self.summary_writer.add_summary(summary, self.nb_ep)
        self.summary_writer.flush()

    def train(self, sess, bootstrap_value):

        # Add the bootstrap value to our experience
        self.rewards_plus = np.asarray(self.rewards_buffer + [bootstrap_value])
        discounted_reward = discount(
            self.rewards_plus, parameters.DISCOUNT)[:-1]

        self.next_values = np.asarray(self.values_buffer[1:] +
                                      [bootstrap_value])
        advantages = self.rewards_buffer + \
            parameters.DISCOUNT * self.next_values - \
            self.values_buffer
        advantages = discount(
            advantages, parameters.GENERALIZED_LAMBDA * parameters.DISCOUNT)


        # Update the global network
        feed_dict = {
            self.network.discounted_reward: discounted_reward,
            self.network.inputs: self.states_buffer,
            self.network.actions: self.actions_buffer,
            self.network.advantages: advantages,
            self.network.state_in: self.initial_lstm_state}
        losses = sess.run([self.network.value_loss,
                           self.network.policy_loss,
                           self.network.entropy,
                           self.network.grad_norm,
                           self.network.apply_grads],
                          feed_dict=feed_dict)

        # Get the losses for tensorboard
        self.value_loss, self.policy_loss, self.entropy = losses[:3]
        self.grad_norm, _ = losses[3:]


        # Reinitialize buffers and variables
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.values_buffer = []

    def work(self, sess, coord):
        print("Running", self.name, end='\n\n')
        self.starting_time = time()
        self.nb_ep = 1

        with sess.as_default(), sess.graph.as_default():

            with coord.stop_on_exception():
                while not coord.should_stop():

                    self.states_buffer = []
                    self.actions_buffer = []
                    self.rewards_buffer = []
                    self.values_buffer = []
                    self.mean_values_buffer = []

                    self.total_steps = 0
                    episode_reward = 0
                    episode_step = 0

                    # Reset the local network to the global
                    sess.run(self.update_local_vars)

                    s = self.env.reset()
                    done = False
                    render = (self.nb_ep % parameters.RENDER_FREQ == 0)
                    if self.worker_index == 1 and render and parameters.DISPLAY:
                        self.env.set_render(True)

                    self.lstm_state = self.network.lstm_state_init
                    self.initial_lstm_state = self.lstm_state

                    while not coord.should_stop() and not done and \
                            episode_step < parameters.MAX_EPISODE_STEP:


                        # Prediction of the policy and the value
                        feed_dict = {self.network.inputs: [s],
                                     self.network.state_in: self.lstm_state}
                        policy, value, self.lstm_state = sess.run(
                            [self.network.policy,
                             self.network.value,
                             self.network.state_out], feed_dict=feed_dict)

                        policy, value = policy[0], value[0][0]

                        if random.random() < self.epsilon:
                            action = random.randint(0, self.action_size - 1)

                        else:
                            # Choose an action according to the policy
                            action = np.random.choice(self.action_size,
                                                      p=policy)

                        s_, r, done, _ = self.env.act(action)

                        # Store the experience
                        self.states_buffer.append(s)
                        self.actions_buffer.append(action)
                        self.rewards_buffer.append(r)
                        self.values_buffer.append(value)
                        self.mean_values_buffer.append(value)
                        episode_reward += r
                        s = s_

                        episode_step += 1
                        self.total_steps += 1

                        # If we have more than MAX_LEN_BUFFER experiences, we
                        # apply the gradients and update the global network,
                        # then we empty the episode buffers
                        if len(self.states_buffer) == parameters.MAX_LEN_BUFFER \
                                and not done:

                            feed_dict = {self.network.inputs: [s],
                                         self.network.state_in: self.lstm_state}
                            bootstrap_value = sess.run(
                                self.network.value,
                                feed_dict=feed_dict)

                            self.train(sess, bootstrap_value)
                            sess.run(self.update_local_vars)
                            self.initial_lstm_state = self.lstm_state

                    if len(self.states_buffer) != 0:
                        if done:
                            bootstrap_value = 0
                        else:
                            feed_dict = {self.network.inputs: [s],
                                         self.network.state_in: self.lstm_state}
                            bootstrap_value = sess.run(
                                self.network.value,
                                feed_dict=feed_dict)
                        self.train(sess, bootstrap_value)

                    if self.epsilon > parameters.EPSILON_STOP:
                        self.epsilon -= parameters.EPSILON_DECAY

                    self.nb_ep += 1

                    if not coord.should_stop():
                        DISPLAYER.add_reward(episode_reward, self.worker_index)

                    if (self.worker_index == 1 and
                            self.nb_ep % parameters.DISP_EP_REWARD_FREQ == 0):
                        print('Episode %2i, Reward: %i, Steps: %i, '
                              'Epsilon: %7.3f' %
                              (self.nb_ep, episode_reward, episode_step,
                               self.epsilon))

                    if (self.worker_index == 1 and
                            self.nb_ep % parameters.SAVE_FREQ == 0):
                        self.save(self.total_steps)

                    if time() - self.starting_time > parameters.LIMIT_RUN_TIME:
                        coord.request_stop()

                    self.env.set_render(False)

            self.summary_writer.close()
            self.env.close()

    def play(self, sess, number_run, path=''):
        print("Playing", self.name, "for", number_run, "runs")

        with sess.as_default(), sess.graph.as_default():

            try:
                for i in range(number_run):

                    # Reset the local network to the global
                    if self.name != 'global':
                        sess.run(self.update_local_vars)

                    s = self.env.reset()
                    episode_reward = 0

                    done = False
                    self.lstm_state = self.network.lstm_state_init

                    while not done:
                        # Prediction of the policy
                        feed_dict = {self.network.inputs: [s],
                                     self.network.state_in: self.lstm_state}
                        policy, self.lstm_state = sess.run(
                            [self.network.policy,
                             self.network.state_out], feed_dict=feed_dict)

                        policy = policy[0]

                        # Choose an action according to the policy
                        action = np.random.choice(self.action_size, p=policy)
                        s, r, done, info = self.env.act(action, path != '')
                        episode_reward += r

                    print("Episode reward :", episode_reward)

                    if path != '':
                        self.env.save_gif(path)

            except KeyboardInterrupt as e:
                pass

            finally:
                print("End of the demo")
                self.env.close()

    def close(self):
        self.env.close()
