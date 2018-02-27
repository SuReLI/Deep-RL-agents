
import tensorflow as tf
import numpy as np

from Model import build_actor, build_critic
from network_utils import copy_vars, get_vars, l2_regularization
from settings import Settings


class Network:
    """
    This class builds four networks :
    - a main actor network that predicts the best possible action given a state
      and a main critic network that predicts the Q-value of this pair (state,
      action)
    - a target actor network and a target critic network which hold a frozen
      copy of the main networks and which are updated periodically
    """

    def __init__(self, sess):
        """
        Creation of the main and target networks and of the tensorflow
        operations to apply a gradient descent and update the target network.

        Args:
            sess: the main tensorflow session in which to create the networks
        """
        print("Creation of the QNetwork...")

        self.sess = sess

        # Batch placeholders
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, *Settings.STATE_SIZE], name='state')
        self.action_ph = tf.placeholder(dtype=tf.float32, shape=[None, Settings.ACTION_SIZE], name='action')
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='reward')
        self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, *Settings.STATE_SIZE], name='next_state')
        self.not_done_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='not_done')
        
        # Turn these in column vector
        self.reward = tf.expand_dims(self.reward_ph, 1)
        self.not_done = tf.expand_dims(self.not_done_ph, 1)

        # Build the networks
        self.build_model()
        self.build_target()
        self.build_update()
        self.build_train_operation()

        print("QNetwork created !")

    def build_model(self):
        """
        Build the main networks.
        
        To improve the critic network, we want to compute the classical TD-error
        TDerr = [ r_t + gamma * Q_target(s_{t+1}, A(s_{t+1})) - Q(s_t, a_t) ]²
        (with A(.) the output of the actor network).

        To improve the actor network, we apply the policy gradient :
        Grad = grad( Q(s_t, A(s_t)) ) * grad( A(s_t) )
        """

        # Compute A(s_t)
        self.actions = build_actor(self.state_ph, trainable=True, scope='actor')

        # Compute Q(s_t, a_t)
        self.q_values_of_given_actions = build_critic(self.state_ph, self.action_ph,
                                                      trainable=True, reuse=False,
                                                      scope='critic')
        # Compute Q(s_t, A(s_t)) with the same network
        self.q_values_of_suggested_actions = build_critic(self.state_ph, self.actions,
                                                          trainable=True, reuse=True,
                                                          scope='critic')

    def build_target(self):
        """
        Build the target networks.
        """
        # Compute A(s_{t+1})
        self.target_next_actions = build_actor(self.next_state_ph,
                                               trainable=False,
                                               scope='target_actor')

        # Compute Q_target( s_{t+1}, A(s_{t+1}) )
        self.q_values_next = build_critic(self.next_state_ph, self.target_next_actions,
                                          trainable=False, reuse=False,
                                          scope='target_critic')

    def build_train_operation(self):
        """
        Define the training operations for the critic and the actor networks.
        The critic is trained with a classical Q-learning (i.e. gradient descent
        with loss = TD-error²).
        The actor is trained according to the policy gradient
        (cf. https://arxiv.org/pdf/1509.02971.pdf).
        """

        # Compute the TD-error
        targets = self.reward + self.not_done * Settings.DISCOUNT * self.q_values_next
        td_errors = targets - self.q_values_of_given_actions

        # Critic loss and optimization
        critic_loss = tf.reduce_mean(tf.square(td_errors))
        critic_loss += l2_regularization(self.critic_vars)
        critic_trainer = tf.train.AdamOptimizer(Settings.CRITIC_LEARNING_RATE)
        self.critic_train = critic_trainer.minimize(critic_loss)

        # Actor loss and optimization
        self.action_grad = tf.gradients(self.q_values_of_suggested_actions, self.actions)[0]
        self.actor_grad = tf.gradients(self.actions, self.actor_vars, -self.action_grad)
        actor_trainer = tf.train.AdamOptimizer(Settings.ACTOR_LEARNING_RATE)
        self.actor_train = actor_trainer.apply_gradients(zip(self.actor_grad, self.actor_vars))

        # Actor loss and optimization
        # actor_loss = -1 * tf.reduce_mean(self.q_values_of_suggested_actions)
        # actor_loss += l2_regularization(self.actor_vars)
        # actor_trainer = tf.train.AdamOptimizer(Settings.ACTOR_LEARNING_RATE)
        # self.actor_train = actor_trainer.minimize(actor_loss,
        #                                              var_list=self.actor_vars)

    def build_update(self):
        """
        Select the network variables and build the operation to copy main
        weights and biases to the target network.
        """
        # Isolate vars for each network
        self.actor_vars = get_vars('actor', trainable=True)
        self.critic_vars = get_vars('critic', trainable=True)
        self.vars = self.actor_vars + self.critic_vars

        self.target_actor_vars = get_vars('target_actor', trainable=False)
        self.target_critic_vars = get_vars('target_critic', trainable=False)
        self.target_vars = self.target_actor_vars + self.target_critic_vars

        # Initial operation to start with target_net == main_net
        self.init_target_op = copy_vars(self.vars, self.target_vars,
                                        1, 'init_target')

        # Update values for target vars towards current actor and critic vars
        self.target_update = copy_vars(self.vars,
                                        self.target_vars,
                                        Settings.UPDATE_TARGET_RATE,
                                        'target_update')
    def init_target(self):
        """
        Wrapper method to initialize the target weights.
        """
        self.sess.run(self.init_target_op)

    def update_target(self):
        """
        Wrapper method to copy the main weights and biases to the target
        network.
        """
        self.sess.run(self.target_update)

    def act(self, state):
        """
        Wrapper method to compute the Q-value distribution given a single state.
        """
        return self.sess.run(self.actions, feed_dict={self.state_ph: [state]})[0]
    
    def train(self, batch):
        """
        Wrapper method to train the network given a minibatch of experiences.
        """
        if len(batch) == 1:
            return

        feed_dict = {self.state_ph: np.stack(batch[:, 0]),
                     self.action_ph: np.stack(batch[:, 1]),
                     self.reward_ph: batch[:, 2],
                     self.next_state_ph: np.stack(batch[:, 3]),
                     self.not_done_ph: batch[:, 4]}

        self.sess.run([self.critic_train, self.actor_train],
                       feed_dict=feed_dict)
