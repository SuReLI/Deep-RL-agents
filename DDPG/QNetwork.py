
import tensorflow as tf
import numpy as np

from Model import *
import settings


class Network:

    def __init__(self, settings, sess):
        print("Creation of the QNetwork...")

        self.settings = settings
        self.sess = sess

        # placeholders
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, *settings.STATE_SIZE], name='a')
        self.action_ph = tf.placeholder(dtype=tf.float32, shape=[None, settings.ACTION_SIZE], name='b')
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='c')
        self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, *settings.STATE_SIZE], name='d')
        self.not_done_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='e')

        self.build_model()
        self.build_target()
        self.build_update()
        self.build_train_operation()

        print("QNetwork created !")

    def build_model(self):

        self.actions = build_actor(self.settings, self.state_ph,
                                   trainable=True, scope='actor')

        self.q_values_of_given_actions = build_critic(
            self.state_ph, self.action_ph, trainable=True, reuse=False, scope='critic')
        self.q_values_of_suggested_actions = build_critic(
            self.state_ph, self.actions, trainable=True, reuse=True, scope='critic')

    def build_target(self):        

        self.target_next_actions = tf.stop_gradient(
            build_actor(self.settings, self.next_state_ph,
                                       trainable=False, scope='target_actor'))

        self.q_values_next = tf.stop_gradient(
            build_critic(self.next_state_ph, self.target_next_actions,
                         trainable=False, reuse=False, scope='target_critic'))

    def build_train_operation(self):

        reward = tf.expand_dims(self.reward_ph, 1)
        not_done = tf.expand_dims(self.not_done_ph, 1)
        targets = reward + not_done * self.settings.DISCOUNT * self.q_values_next

        # 1-step temporal difference errors
        td_errors = targets - self.q_values_of_given_actions

        # Critic loss and optimization
        critic_loss = tf.reduce_mean(tf.square(td_errors))
        critic_loss += l2_regularization(self.critic_vars)
        critic_trainer = tf.train.AdamOptimizer(self.settings.CRITIC_LEARNING_RATE)
        self.critic_train = critic_trainer.minimize(critic_loss)

        # Actor loss and optimization
        # self.action_grad = tf.gradients(self.q_values_of_suggested_actions, self.actions)[0]
        # self.actor_grad = tf.gradients(self.actions, self.actor_vars, -self.action_grad)
        # actor_trainer = tf.train.AdamOptimizer(self.settings.ACTOR_LEARNING_RATE)
        # self.actor_train_op = actor_trainer.apply_gradients(zip(self.actor_grad, self.actor_vars))

        # Actor loss and optimization
        actor_loss = -1 * tf.reduce_mean(self.q_values_of_suggested_actions)
        actor_loss += l2_regularization(self.actor_vars)
        actor_trainer = tf.train.AdamOptimizer(self.settings.ACTOR_LEARNING_RATE)
        self.actor_train = actor_trainer.minimize(actor_loss,
                                                     var_list=self.actor_vars)

    def build_update(self):

        # Isolate vars for each network
        self.actor_vars = get_vars('actor', trainable=True)
        self.critic_vars = get_vars('critic', trainable=True)
        self.vars = self.actor_vars + self.critic_vars

        self.target_actor_vars = get_vars('target_actor', trainable=False)
        self.target_critic_vars = get_vars('target_critic', trainable=False)
        self.target_vars = self.target_actor_vars + self.target_critic_vars

        # Initialize target critic vars to critic vars
        self.target_init = copy_vars(self.vars,
                                     self.target_vars,
                                     1, 'init_target')

        # Update values for target vars towards current actor and critic vars
        self.update_targets = copy_vars(self.vars,
                                        self.target_vars,
                                        self.settings.UPDATE_TARGET_RATE,
                                        'update_targets')

    def init_target_update(self):
        _ = self.sess.run(self.target_init)

    def target_update(self):
        _ = self.sess.run(self.update_targets)

    def train(self, batch):

        if len(batch) == 1:
            return

        state = batch[:, 0]
        action = batch[:, 1]
        reward = batch[:, 2]
        next_state = batch[:, 3]
        not_done = batch[:, 4]

        feed_dict = {self.state_ph: np.stack(state),
                     self.action_ph: np.stack(action),
                     self.reward_ph: reward,
                     self.next_state_ph: np.stack(next_state),
                     self.not_done_ph: not_done}
        _, _ = self.sess.run([self.critic_train, self.actor_train],
                             feed_dict=feed_dict)
