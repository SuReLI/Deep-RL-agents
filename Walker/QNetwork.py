
import tensorflow as tf

import parameters


class Network:

    def __init__(self, state_size, action_size, low_bound, high_bound):

        self.state_size = state_size
        self.action_size = action_size

        # placeholders
        self.state_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.state_size])
        self.action_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.action_size])
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        self.next_state_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.state_size])
        self.is_not_terminal_ph = tf.placeholder(
            dtype=tf.float32, shape=[None])

        # Actor definition :
        def generate_actor_network(states, trainable, reuse):
            hidden = tf.layers.dense(states, 64,
                                     trainable=trainable, reuse=reuse,
                                     activation=tf.nn.relu, name='dense')
            hidden_2 = tf.layers.dense(hidden, 64,
                                       trainable=trainable, reuse=reuse,
                                       activation=tf.nn.relu, name='dense_1')
            hidden_3 = tf.layers.dense(hidden_2, 64,
                                       trainable=trainable, reuse=reuse,
                                       activation=tf.nn.relu, name='dense_2')
            actions_unscaled = tf.layers.dense(hidden_3, self.action_size,
                                               trainable=trainable, reuse=reuse,
                                               name='dense_3')
            # bound the actions to the valid range
            valid_range = high_bound - low_bound
            actions = low_bound + tf.nn.sigmoid(actions_unscaled) * valid_range
            return actions

        # Main actor network
        with tf.variable_scope('actor'):
            self.actions = generate_actor_network(self.state_ph,
                                                  trainable=True, reuse=False)

        # Target actor network
        with tf.variable_scope('slow_target_actor', reuse=False):
            self.slow_target_next_actions = tf.stop_gradient(
                generate_actor_network(self.next_state_ph,
                                       trainable=False, reuse=False))

        # Critic definition :
        def generate_critic_network(states, actions, trainable, reuse):
            state_action = tf.concat([states, actions], axis=1)
            hidden = tf.layers.dense(state_action, 64,
                                     trainable=trainable, reuse=reuse,
                                     activation=tf.nn.relu, name='dense')
            hidden_2 = tf.layers.dense(hidden, 64,
                                       trainable=trainable, reuse=reuse,
                                       activation=tf.nn.relu, name='dense_1')
            hidden_3 = tf.layers.dense(hidden_2, 64,
                                       trainable=trainable, reuse=reuse,
                                       activation=tf.nn.relu, name='dense_2')
            q_values = tf.layers.dense(hidden_3, 1,
                                       trainable=trainable, reuse=reuse,
                                       name='dense_3')
            return q_values

        with tf.variable_scope('critic') as scope:
            # Critic applied to state_ph and a given action (to train critic)
            self.q_values_of_given_actions = generate_critic_network(
                self.state_ph, self.action_ph, trainable=True, reuse=False)
            # Critic applied to state_ph and the current policy's outputted
            # actions for state_ph (to train actor)
            self.q_values_of_suggested_actions = generate_critic_network(
                self.state_ph, self.actions, trainable=True, reuse=True)

        # slow target critic network
        with tf.variable_scope('slow_target_critic', reuse=False):
            # Slow target critic applied to slow target actor's outputted
            # actions for next_state_ph (to train critic)
            self.slow_q_values_next = tf.stop_gradient(generate_critic_network(
                self.next_state_ph, self.slow_target_next_actions,
                trainable=False, reuse=False))

        # isolate vars for each network
        self.actor_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        self.slow_target_actor_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_actor')
        self.critic_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        self.slow_target_critic_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_critic')

        # update values for slowly-changing targets towards current actor and
        # critic
        update_slow_target_ops = []
        for i, slow_target_actor_var in enumerate(self.slow_target_actor_vars):
            update_slow_target_actor_op = slow_target_actor_var.assign(
                parameters.UPDATE_TARGET_RATE * self.actor_vars[i] +
                (1 - parameters.UPDATE_TARGET_RATE) * slow_target_actor_var)
            update_slow_target_ops.append(update_slow_target_actor_op)

        for i, slow_target_var in enumerate(self.slow_target_critic_vars):
            update_slow_target_critic_op = slow_target_var.assign(
                parameters.UPDATE_TARGET_RATE * self.critic_vars[i] +
                (1 - parameters.UPDATE_TARGET_RATE) * slow_target_var)
            update_slow_target_ops.append(update_slow_target_critic_op)

        self.update_slow_targets_op = tf.group(*update_slow_target_ops,
                                               name='update_slow_targets')

        # One step TD targets y_i for (s,a) from experience replay
        # = r_i + parameters.DISCOUNT*Q_slow(s',mu_slow(s')) if s' is not terminal
        # = r_i if s' terminal
        targets = tf.expand_dims(self.reward_ph, 1) + \
            tf.expand_dims(self.is_not_terminal_ph, 1) * parameters.DISCOUNT * \
            self.slow_q_values_next

        # 1-step temporal difference errors
        td_errors = targets - self.q_values_of_given_actions

        # Critic loss and optimization
        critic_loss = tf.reduce_mean(tf.square(td_errors))
        for var in self.critic_vars:
            if not 'bias' in var.name:
                critic_loss += 1e-6 * 0.5 * tf.nn.l2_loss(var)

        critic_trainer = tf.train.AdamOptimizer(
            parameters.CRITIC_LEARNING_RATE)
        self.critic_train_op = critic_trainer.minimize(critic_loss)

        # Actor loss and optimization
        actor_loss = -1 * tf.reduce_mean(self.q_values_of_suggested_actions)
        for var in self.actor_vars:
            if not 'bias' in var.name:
                actor_loss += 1e-6 * 0.5 * tf.nn.l2_loss(var)

        actor_trainer = tf.train.AdamOptimizer(parameters.ACTOR_LEARNING_RATE)

        self.actor_train_op = actor_trainer.minimize(actor_loss,
                                                     var_list=self.actor_vars)
