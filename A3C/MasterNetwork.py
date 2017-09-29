
import tensorflow as tf
import numpy as np
import Model
import Agent


class Network:

    def __init__(self, env, model):

        self.master_agent = Agent(worker_index=0, render=True)
        state_dim = self.master_agent.env.get_state_dims()

        assert type(state_dim) == tuple

        self.state = tf.placeholder(tf.float32, [None, *state_dim],
                                    name='Input_state')

        self.action_logits, self.value = Model(model, self.state)
        self.action = tf.squeeze(tf.multnomial(
            self.action_logits - tf.reduce_max(self.action_logits, 1, keep_dims=True), 1))
