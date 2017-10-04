
import tensorflow as tf
import numpy as np
from Environment import Environment
from MasterNetwork import Network
import parameters
import Saver
import scipy.signal


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

    def __init__(self, worker_index, render=False, master=False):
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

        self.rewards = []
        self.mean_values = []

    def update_global_network(self, sess, s, lstm_state):
        bootstrap_value = sess.run(
            self.network.value,
            feed_dict={self.network.inputs: [s],
                       self.network.state_in: lstm_state})

        # Add the bootstrap value to our experience
        self.rewards_plus = np.asarray(self.rewards_buffer + [bootstrap_value])
        discounted_reward = discount(self.rewards_plus)[:-1]

        self.values_plus = np.asarray(self.values_buffer + [bootstrap_value])
        advantage = self.rewards_plus[:-1] + \
            parameters.DISCOUNT * self.values_plus[1:] - \
            self.values_plus[:-1]
        advantage = discount(advantage)

        # Update the global network
        feed_dict = {
            self.network.discounted_reward: discounted_reward,
            self.network.inputs: self.states_buffer,
            self.network.actions: self.actions_buffer,
            self.network.advantage: advantage,
            self.network.state_in: lstm_state}
        losses = sess.run([self.network.value_loss,
                           self.network.policy_loss,
                           self.network.entropy,
                           self.network.grad_norm,
                           self.network.state_out,
                           self.network.apply_grads],
                          feed_dict=feed_dict)

        # Get the losses for tensorboard (TO DO)
        value_loss, policy_loss, entropy = losses[:3]
        grad_norm, lstm_state, _ = losses[3:]

        # Reinitialize buffers and variables
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.values_buffer = []
        sess.run(self.update_local_vars)

        return lstm_state

    def work(self, sess, coord):
        print("Running", self.name, end='\n\n')
        total_steps = 0

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
                    lstm_state = self.network.lstm_state_init

                    while not coord.should_stop() and not done and \
                            episode_step < parameters.MAX_EPISODE_STEP:
                        # Prediction of the policy and the value
                        feed_dict = {self.network.inputs: [s],
                                     self.network.state_in: lstm_state}
                        policy, value, lstm_state = sess.run(
                            [self.network.policy,
                             self.network.value,
                             self.network.state_out], feed_dict=feed_dict)

                        policy, value = policy[0], value[0][0]

                        # Choose an action according to the policy
                        action = np.random.choice(self.action_size, p=policy)
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
                        total_steps += 1

                        # If we have more than MAX_LEN_BUFFER experiences, we
                        # apply the gradients and update the global network,
                        # then we empty the episode buffers
                        if len(self.states_buffer) == parameters.MAX_LEN_BUFFER \
                                and not done:
                            lstm_state = self.update_global_network(sess,
                                                                    s,
                                                                    lstm_state)

                    # print("Episode reward of {} : {}".format(self.name, reward))
                    self.rewards.append(reward)
                    Saver.add_results(reward)

                    self.mean_values.append(np.mean(self.mean_values_buffer))

                    if len(self.states_buffer) != 0:
                        lstm_state = self.update_global_network(sess,
                                                                s,
                                                                lstm_state)
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
                    lstm_state = self.network.lstm_state_init

                    while not done:
                        # Prediction of the policy and the value
                        feed_dict = {self.network.inputs: [s],
                                     self.network.state_in: lstm_state}
                        policy, value, lstm_state = sess.run(
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
