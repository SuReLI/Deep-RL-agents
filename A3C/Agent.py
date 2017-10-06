
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
    return scipy.signal.lfilter([1],
                                [1, -parameters.DISCOUNT],
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

        self.network = Network(self.state_size, self.action_size, self.name)
        self.update_local_vars = update_target_graph('global', self.name)

        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.values_buffer = []

        if self.name != 'global':
            self.summary_writer = tf.summary.FileWriter("results/" + self.name,
                                                        sess.graph)

    def update_global_network(self, sess, bootstrap_value):

        # Add the bootstrap value to our experience
        self.rewards_plus = np.asarray(self.rewards_buffer + [bootstrap_value])
        discounted_reward = discount(self.rewards_plus)[:-1]

        self.next_values = np.asarray(self.values_buffer[1:] +
                                      [bootstrap_value])
        advantages = self.rewards_buffer + \
            parameters.DISCOUNT * self.next_values - \
            self.values_buffer
        advantages = discount(advantages)

        # Update the global network
        feed_dict = {
            self.network.discounted_reward: discounted_reward,
            self.network.inputs: self.states_buffer,
            self.network.actions: self.actions_buffer,
            self.network.advantages: advantages,
            self.network.state_in: self.lstm_state}
        losses = sess.run([self.network.value_loss,
                           self.network.policy_loss,
                           self.network.entropy,
                           self.network.grad_norm,
                           self.network.state_out,
                           self.network.apply_grads],
                          feed_dict=feed_dict)

        # Get the losses for tensorboard
        value_loss, policy_loss, entropy = losses[:3]
        grad_norm, self.lstm_state, _ = losses[3:]

        # Save summary statistics
        summary = tf.Summary()
        summary.value.add(tag='Perf/Reward',
                          simple_value=np.mean(self.rewards_plus))
        summary.value.add(tag='Perf/Value',
                          simple_value=np.mean(self.next_values))
        summary.value.add(tag='Perf/Advantage',
                          simple_value=np.mean(advantages))
        summary.value.add(tag='Losses/Value',
                          simple_value=value_loss)
        summary.value.add(tag='Losses/Policy',
                          simple_value=policy_loss)
        summary.value.add(tag='Losses/Entropy',
                          simple_value=entropy)
        summary.value.add(tag='Losses/Grad Norm',
                          simple_value=grad_norm)
        self.summary_writer.add_summary(summary, self.total_steps)
        self.summary_writer.flush()

        # Reinitialize buffers and variables
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.values_buffer = []

    def work(self, sess, coord):
        print("Running", self.name, end='\n\n')
        self.total_steps = 0

        with sess.as_default(), sess.graph.as_default():

            with coord.stop_on_exception():
                while not coord.should_stop():
                    self.states_buffer = []
                    self.actions_buffer = []
                    self.rewards_buffer = []
                    self.values_buffer = []
                    self.mean_values_buffer = []
                    reward = 0
                    episode_step = 0

                    # Reset the local network to the global
                    sess.run(self.update_local_vars)

                    s = self.env.reset()
                    done = False
                    self.lstm_state = self.network.lstm_state_init

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

                        if Agent.epsilon > parameters.EPSILON_STOP:
                            Agent.epsilon -= Agent.epsilon_decay

                        if random.random() < Agent.epsilon:
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
                        reward += r
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
                            self.update_global_network(sess, bootstrap_value)
                            sess.run(self.update_local_vars)

                    # print("Episode reward of {} : {}".format(self.name,
                    #                                          reward))
                    DISPLAYER.add_reward(reward, self.worker_index)

                    if len(self.states_buffer) != 0:
                        self.update_global_network(sess, 0)
            self.summary_writer.close()
            self.env.close()

    def play(self, sess, number_run):
        print("Playing", self.name, "for", number_run, "runs")

        with sess.as_default(), sess.graph.as_default():

            try:
                # Reset the local network to the global
                sess.run(self.update_local_vars)

                for _ in range(number_run):

                    s = self.env.reset()
                    reward = 0

                    done = False
                    self.lstm_state = self.network.lstm_state_init

                    while not done:
                        # Prediction of the policy and the value
                        feed_dict = {self.network.inputs: [s],
                                     self.network.state_in: self.lstm_state}
                        policy, value, self.lstm_state = sess.run(
                            [self.network.policy,
                             self.network.value,
                             self.network.state_out], feed_dict=feed_dict)

                        policy, value = policy[0], value[0][0]

                        # Choose an action according to the policy
                        action = np.random.choice(self.action_size, p=policy)
                        s, r, done, _ = self.env.act(action)
                        reward += r

                    print("Episode reward :", reward)

            except KeyboardInterrupt as e:
                pass

            finally:
                print("End of the demo")
                self.env.close()
