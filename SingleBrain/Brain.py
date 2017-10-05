

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, LSTM, Reshape
from keras import backend as K

import threading
import numpy as np
from time import sleep

from parameters import LOSS_VALUE_REG, LOSS_ENTROPY_REG
from parameters import LEARNING_RATE, MIN_BATCH, GAMMA_N


class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self, state_size, action_size):
        print("Initializing the brain")

        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.state_size = state_size
        self.action_size = action_size
        self.none_state = np.zeros(self.state_size)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

    def _build_model(self):

        l_input = Input(batch_shape=(None, *self.state_size))
        l_dense = Dense(16, activation='relu')(l_input)

        conv1 = Conv2D(filters=32, kernel_size=(6, 6),
                       strides=(2, 2), activation='relu')(l_input)
        conv2 = Conv2D(filters=64, kernel_size=(6, 6),
                       strides=(2, 2), activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(6, 6),
                       strides=(2, 2), activation='relu')(conv2)
        conv4 = Conv2D(filters=64, kernel_size=(6, 6),
                       strides=(2, 2), activation='relu')(conv3)

        l_reshape = Reshape((-1, np.prod(conv4.shape.as_list()[1:])))(conv4)
        l_lstm = LSTM(256, input_shape=l_reshape.shape[1:])(l_reshape)

        out_actions = Dense(self.action_size, activation='softmax')(l_lstm)
        out_value = Dense(1, activation='linear')(l_lstm)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()  # have to initialize before threading

        return model

    def _build_graph(self, model):
        state = tf.placeholder(tf.float32, shape=(None, *self.state_size))
        action = tf.placeholder(tf.float32, shape=(None, self.action_size))
        # not immediate, but discounted n step reward
        reward = tf.placeholder(tf.float32, shape=(None, 1))

        pi, value = model(state)

        log_prob = tf.log(tf.reduce_sum(
            pi * action, axis=1, keep_dims=True) + 1e-10)
        advantage = reward - value

        # maximize policy
        loss_policy = - log_prob * tf.stop_gradient(advantage)
        # minimize value error
        loss_value = LOSS_VALUE_REG * tf.square(advantage)
        # maximize entropy (regularization)
        entropy = LOSS_ENTROPY_REG * \
            tf.reduce_sum(pi * tf.log(pi + 1e-10), axis=1, keep_dims=True)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return state, action, reward, minimize

    def optimize(self):
        """Compute the loss and minimize it from the experience buffer"""

        if len(self.train_queue[0]) < MIN_BATCH:
            # We don't have enough experience to train
            sleep(0)   # yield
            return

        with self.lock_queue:
            # more thread could have passed without lock
            if len(self.train_queue[0]) < MIN_BATCH:
                return        # we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.array(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.array(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5 * MIN_BATCH:
            print("Optimizer alert! Minimizing batch of %d" % len(s))

        pi, v = self.predict(s_)
        r = r + GAMMA_N * v * s_mask    # set v to 0 where s_ is terminal state

        state, action, reward, minimize = self.graph
        self.session.run(minimize, feed_dict={state: s, action: a, reward: r})

    def train_push(self, s, a, r, s_):
        """Add a set of experiences to the experience buffer"""

        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(self.none_state)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        """Predict the policy and the value from an input state"""
        with self.default_graph.as_default():
            pi, value = self.model.predict(s)
            return pi, value

    def save(self):
        self.model.save_weights("Model.h5")

    def load(self):
        self.model.load_weights("Model.h5")


class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self, brain):
        threading.Thread.__init__(self)
        self.brain = brain

    def run(self):
        while not self.stop_signal:
            self.brain.optimize()

    def stop(self):
        self.stop_signal = True
