#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 15:06:04 2017

@author: valentin
"""

import numpy as np
import tensorflow as tf

import gym
import threading
import time
import random
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K

# -- constants
ENV = 'CartPole-v0'

RUN_TIME = 60
THREADS = 8
OPTIMIZERS = 4
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPSILON_START = 0.8
EPSILON_STOP = .01
EPSILON_STEPS = 100000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5         # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient


# -------------------- BRAIN ---------------------------
class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()   # avoid modifications

        self.rewards = [[] for i in range(THREADS+1)]
        self.sequential_rewards = []

    def _build_model(self):

        l_input = Input(batch_shape=(None, NUM_STATE))
        l_dense = Dense(16, activation='relu')(l_input)

        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()  # have to initialize before threading

        return model

    def _build_graph(self, model):
        state = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
        action = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        # not immediate, but discounted n step reward
        reward = tf.placeholder(tf.float32, shape=(None, 1))

        pi, value = model(state)

        log_prob = tf.log(tf.reduce_sum(
            pi * action, axis=1, keep_dims=True) + 1e-10)
        advantage = reward - value

        # maximize policy
        loss_policy = - log_prob * tf.stop_gradient(advantage)
        # minimize value error
        loss_value = LOSS_V * tf.square(advantage)
        # maximize entropy (regularization)
        entropy = LOSS_ENTROPY * \
            tf.reduce_sum(pi * tf.log(pi + 1e-10), axis=1, keep_dims=True)

        losstateotal = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(losstateotal)

        return state, action, reward, minimize

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)   # yield
            return

        with self.lock_queue:
            # more thread could have passed without lock
            if len(self.train_queue[0]) < MIN_BATCH:
                return        # we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5 * MIN_BATCH:
            print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask    # set v to 0 where s_ is terminal state

        state, action, reward, minimize = self.graph
        self.session.run(minimize, feed_dict={state: s, action: a, reward: r})

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            pi, value = self.model.predict(s)
            return pi, value

    def predict_p(self, s):
        with self.default_graph.as_default():
            pi, value = self.model.predict(s)
            return pi

    def predict_v(self, s):
        with self.default_graph.as_default():
            pi, value = self.model.predict(s)
            return value

    def add_reward(self, R, agent):
        self.rewards[agent].append(R)
        if agent != 0:
            self.sequential_rewards.append(R)


# -------------------- AGENT ---------------------------
frames = 0


class Agent:

    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

        self.memory = []    # used for n_step return
        self.R = 0.
        self.rewards = []

    def save(self, R):
        self.rewards.append(R)

    def disp(self):
        plt.plot(self.rewards)

    def getEpsilon(self):
        if(frames >= self.eps_steps):
            return self.eps_end
        else:
            # linearly interpolate
            return self.eps_start + frames * (self.eps_end - self.eps_start) \
                    / self.eps_steps

    def act(self, s):
        eps = self.getEpsilon()
        global frames
        frames += 1

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)

        else:
            s = np.array([s])
            p = brain.predict_p(s)[0]

            # a = np.argmax(p)
            a = np.random.choice(NUM_ACTIONS, p=p)

            return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        # turn action into one-hot representation
        a_onehot = np.zeros(NUM_ACTIONS)
        a_onehot[a] = 1

        self.memory.append((s, a_onehot, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)


# -------------------- ENVIRONMENT ---------------------
class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, n_agent, eps_start=EPSILON_START,
                 eps_end=EPSILON_STOP, eps_steps=EPSILON_STEPS):
        threading.Thread.__init__(self)

        self.env = gym.make(ENV)
        self.agent = Agent(eps_start, eps_end, eps_steps)
        self.n_agent = n_agent

    def runEpisode(self, render=False):
        s = self.env.reset()

        R = 0
        while True:
            time.sleep(THREAD_DELAY)  # yield

            if render:
                self.env.render()

            a = self.agent.act(s)
            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None

            self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                break

        if not self.stop_signal:
            # print("Total R:", R)
            self.agent.save(R)
            brain.add_reward(R, self.n_agent)

    def run(self, render=False):
        while not self.stop_signal:
            self.runEpisode(render)

    def stop(self):
        self.stop_signal = True
        self.env.render(close=True)
        self.env.close()


# -------------------- OPTIMIZER ---------------------
class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True


# -------------------- MAIN ----------------------------
def disp():

    plt.plot(brain.sequential_rewards)
    x = [np.mean(brain.sequential_rewards[max(i-100, 1):i])
         for i in range(2, len(brain.sequential_rewards))]
    plt.plot(x)
    env_test.agent.disp()


env_test = Environment(0, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape[0]
NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(NUM_STATE)

brain = Brain()  # brain is global in A3C

envs = [Environment(i+1) for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()

print("Training finished")
try:
    env_test.run(render=True)
except KeyboardInterrupt as e:
    print("End of the session")
    env_test.env.render(close=True)
