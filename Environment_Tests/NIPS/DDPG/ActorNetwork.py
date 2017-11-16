
import tensorflow as tf
import numpy as np

import parameters


class ActorNetwork:

    def __init__(self, sess, state_size, action_size, low_bound, high_bound):

        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.low_bound = low_bound
        self.high_bound = high_bound

        # Create actor network
        self.state_input, self.action_output, self.vars = self.create_network(
            'main_actor')

        # Create target actor network
        self.target_state_input, self.target_action_output, self.target_vars = self.create_network(
            'target_actor')

        # Define training rules
        self.create_training_method()

        # Define update target rules
        self.create_update_method()

        self.sess.run(tf.global_variables_initializer())

        # WARNING : this update must be done with tau = 1 (target = main first)
        self.update_target()

        if parameters.LOAD:
            self.load_network()

    def create_network(self, scope):

        state_input = tf.placeholder(tf.float32, [None] + list(self.state_size))

        with tf.variable_scope(scope):
            hidden = tf.layers.dense(state_input, 8,
                                     activation=tf.nn.relu,
                                     name=scope + '_dense')

            hidden_2 = tf.layers.dense(hidden, 8,
                                       activation=tf.nn.relu,
                                       name=scope + '_dense_1')

            hidden_3 = tf.layers.dense(hidden_2, 8,
                                       activation=tf.nn.relu,
                                       name=scope + '_dense_2')

            actions_unscaled = tf.layers.dense(hidden_3, self.action_size,
                                               name=scope + '_dense_3')
        # Bound the actions to the valid range
        valid_range = self.high_bound - self.low_bound
        action_output = self.low_bound + \
            tf.nn.sigmoid(actions_unscaled) * valid_range

        actor_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        return state_input, action_output, actor_vars

    def create_training_method(self):
        self.q_gradient_input = tf.placeholder(
            tf.float32, [None, self.action_size])

        self.parameters_gradients = tf.gradients(
            self.action_output, self.vars, -self.q_gradient_input)

        trainer = tf.train.AdamOptimizer(parameters.ACTOR_LEARNING_RATE)
        self.optimizer = trainer.apply_gradients(
            zip(self.parameters_gradients, self.vars))

    def create_update_method(self):
        tau = parameters.UPDATE_TARGET_RATE
        update_target_ops = []
        for target_var, main_var in zip(self.target_vars, self.vars):
            op = target_var.assign(tau * main_var + (1 - tau) * target_var)
            update_target_ops.append(op)

        self.target_update = tf.group(*update_target_ops,
                                      name='update_actor_target')

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer,
                      feed_dict={self.q_gradient_input: q_gradient_batch,
                                 self.state_input: state_batch}
                      )

    def actions(self, state_batch):
        return self.sess.run(self.action_output,
                             feed_dict={self.state_input: state_batch}
                             )

    def action(self, state):
        return self.sess.run(self.action_output,
                             feed_dict={self.state_input: [state]}
                             )[0]

    def target_actions(self, state_batch):
        return self.sess.run(self.target_action_output,
                             feed_dict={self.target_state_input: state_batch}
                             )

    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save_network(self, time_step):
        print('Save actor-network...', time_step)
        self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network',
                        global_step=time_step)
