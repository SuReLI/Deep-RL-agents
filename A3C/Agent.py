
import tensorflow as tf
import numpy as np
from Environment import Environment
from parameters import ENV


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

    def __init__(self, worker_index, render=False):
        print("Initialization of the agent", str(worker_index))

        self.worker_index = worker_index
        self.name = 'Worker_' + str(worker_index)

        self.env = Environment()
        state_size = self.env.get_state_size()
        action_size = self.env.get_action_size()

        self.network = Network(state_size, action_size, self.name)
        self.update_local_vars = update_target_graph('global', self.name)

        self.rewards = []
        self.mean_values = []

    def work(self, sess, coord):
        print("Starting", self.name)
        total_steps = 0

        with sess.as_default(), sess.graph.as_default():

            while not coord.should_stop():
            	experience_buffer = []
            	values_buffer = []
            	reward = 0
            	episode_step = 0

                # Reset the local network to the global
                sess.run(self.update_local_vars)

                s = self.env.reset()
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

                    # Choose an action according to the policy
                    action = np.random.choice(action_size, p=policy)
                    s_, r, done, _ = self.env.act(action)

                    experience_buffer.append([s, a, r, s_, value])
                    values_buffer.append(v)
                    reward += r
                    s = s_

                    episode_step += 1
                    total_steps += 1

                    if len(experience_buffer) == parameters.MAX_LEN_BUFFER and \
                    		not done:
                    	bootstrap_value = sess.run(self.network.value,
                    		feed_dict={self.network.inputs: [s],
                    				   self.network.state_in: lstm_state})
                    	self.train()

                    	experience_buffer = []
                    	sess.run(self.update_local_vars)

                self.rewards.append(reward)
                self.mean_values.append(np.mean(values_buffer))

                self.train()

                
