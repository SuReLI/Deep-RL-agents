
import tensorflow as tf
import numpy as np

import parameters


class CriticNetwork:

    def __init__(self, sess, state_size, action_size):

        self.time_step = 0
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size

        # create Q-Network
        self.state_input,\
            self.action_input,\
            self.q_value_output,\
            self.vars = self.create_q_network('main_critic')

        # create target Q-Network (the same structure with q network)
        self.target_state_input,\
            self.target_action_input,\
            self.target_q_value_output,\
            self.target_vars = self.create_q_network('target_critic')

        # Define training rules
        self.create_training_method()

        # Define update target rules
        self.create_update_method()

        self.sess.run(tf.global_variables_initializer())

        # WARNING : this update must be done with tau = 1 (target = main first)
        self.update_target()

        if parameters.LOAD:
            self.load_network()

    def create_q_network(self, scope):

        state_input = tf.placeholder(tf.float32, [None, *self.state_size])
        action_input = tf.placeholder(tf.float32, [None, self.action_size])

        with tf.variable_scope(scope):
            state_action = tf.concat([state_input, action_input], axis=1)
            hidden = tf.layers.dense(state_action, 8,
                                     activation=tf.nn.relu,
                                     name=scope + '_dense')
            hidden_2 = tf.layers.dense(hidden, 8,
                                       activation=tf.nn.relu,
                                       name=scope + '_dense_1')
            hidden_3 = tf.layers.dense(hidden_2, 8,
                                       activation=tf.nn.relu,
                                       name=scope + '_dense_2')
            q_value_output = tf.layers.dense(hidden_3, 1,
                                             name=scope + '_dense_3')

        critic_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        return state_input, action_input, q_value_output, critic_vars

    def create_training_method(self):
        # Define training optimizer
        self.y_input = tf.placeholder("float", [None, 1])
        weight_decay = tf.add_n([parameters.CRITIC_REG * tf.nn.l2_loss(var)
                                 for var in self.vars])
        self.cost = tf.reduce_mean(
            tf.square(self.y_input - self.q_value_output)) + weight_decay

        trainer = tf.train.AdamOptimizer(parameters.CRITIC_LEARNING_RATE)
        self.optimizer = trainer.minimize(self.cost)
        self.action_gradients = tf.gradients(self.q_value_output,
                                             self.action_input)

    def create_update_method(self):
        tau = parameters.UPDATE_TARGET_RATE
        update_target_ops = []
        for target_var, main_var in zip(self.target_vars, self.vars):
            op = target_var.assign(tau * main_var + (1 - tau) * target_var)
            update_target_ops.append(op)

        self.target_update = tf.group(*update_target_ops,
                                      name='update_critic_target')

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, y_batch, state_batch, action_batch):
        self.time_step += 1
        self.sess.run(self.optimizer,
                      feed_dict={self.y_input: y_batch,
                                 self.state_input: state_batch,
                                 self.action_input: action_batch
                                 })

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients,
                             feed_dict={
                                 self.state_input: state_batch,
                                 self.action_input: action_batch
                             })[0]

    def target_q(self, state_batch, action_batch):
        return self.sess.run(self.target_q_value_output,
                             feed_dict={
                                 self.target_state_input: state_batch,
                                 self.target_action_input: action_batch
                             })

    def q_value(self, state_batch, action_batch):
        return self.sess.run(self.q_value_output,
                             feed_dict={
                                 self.state_input: state_batch,
                                 self.action_input: action_batch})

    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_critic_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save_network(self, time_step):
        print('save critic-network...', time_step)
        self.saver.save(self.sess, 'saved_critic_networks/' + 'critic-network',
                        global_step=time_step)
