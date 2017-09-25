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
from keras.layers import Input, Dense, Lambda, Reshape
from keras import backend as K

# -- constants
ENV = 'BipedalWalker-v2'

RUN_TIME = 10000
THREADS = 16
OPTIMIZERS = 8
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPSILON_START = 0.8
EPSILON_STOP = .01
EPSILON_STEPS = 10000000

MIN_BATCH = 64
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

        # self.default_graph.finalize()   # avoid modifications

        self.rewards = [[] for i in range(THREADS+1)]
        self.sequential_rewards = []

    def _build_model(self):

        l_input = Input(batch_shape=(None, NUM_STATE))
        l_dense = Dense(64, activation='relu')(l_input)

        # mu_sigma = Dense(2*NUM_ACTIONS, activation='linear')(l_dense)
        # mu_sigma = Reshape([NUM_ACTIONS, 2])(mu_sigma)

        # def generate_normal_sample(mu_sigma):
        # mu = mu_sigma[:, :, 0]
        # sigma = mu_sigma[:, :, 1]
        # actions = K.random_normal([4], mu, sigma)
        # out_actions = K.clip(actions, -1, 1)
        # return mu_sigma[:, :NUM_ACTIONS, 0]

        # out_actions = Lambda(generate_normal_sample)(mu_sigma)
#        print("HERE ! "*10)
#        print(out_actions.shape)
        out_mu = Dense(NUM_ACTIONS, activation='tanh')(l_dense)
        out_sigma = Dense(NUM_ACTIONS, activation='tanh')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)

        model = Model(inputs=[l_input], outputs=[out_mu, out_sigma, out_value])
        model._make_predict_function()  # have to initialize before threading

        return model

    def _build_graph(self, model):
        state = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
        action = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        # not immediate, but discounted n step reward
        reward = tf.placeholder(tf.float32, shape=(None, 1))

        mu, sigma, value = model(state)
        sigma = tf.nn.softplus(sigma) + 1e-5
        advantage = reward - value

        pi = tf.contrib.distributions.Normal(mu, sigma)
        action_sample = pi.sample(1)
        action_sample = tf.clip_by_value(action_sample, -1, 1)

        # maximize policy
        loss_policy = -pi.log_prob(action_sample) * tf.stop_gradient(advantage)

        # maximize entropy (regularization)
        entropy = LOSS_ENTROPY * pi.entropy()

        # minimize value error
        loss_value = LOSS_V * tf.square(advantage)
#
#        log_prob = tf.log(tf.reduce_sum(
#            pi * action, axis=1, keep_dims=True) + 1e-10)
#
#        # maximize policy
#        loss_policy = - log_prob * tf.stop_gradient(advantage)
#
#        # maximize entropy (regularization)
#        entropy = LOSS_ENTROPY * \
#            tf.reduce_sum(pi * tf.log(pi + 1e-10), axis=1, keep_dims=True)

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
            mu, sigma, value = self.model.predict(s)
            sigma = np.log(1+np.exp(sigma)) + 1e-5
            return mu, sigma+1, value

    def predict_p(self, s):
        with self.default_graph.as_default():
            mu, sigma, value = self.model.predict(s)
            sigma = np.log(1+np.exp(sigma)) + 1e-5
            return mu, sigma+1

    def predict_v(self, s):
        with self.default_graph.as_default():
            mu, sigma, value = self.model.predict(s)
            sigma = np.log(1+np.exp(sigma)) + 1e-5
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
        self.mus = []
        self.sigmas = []

    def save(self, R):
        self.rewards.append(R)

    def disp(self):
        plt.plot(self.rewards)
        plt.plot(self.mus)
        plt.plot(self.sigmas)

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
            mu = np.random.uniform(-1, 1, 4)
            sigma = np.random.uniform(0.1, 1.9, 4)
            a = np.random.normal(mu, sigma)
            return a

        else:
            s = np.array([s])
            mu, sigma = brain.predict_p(s)
            mu = mu[0]
            sigma = sigma[0]
            self.mus.append(mu)
            self.sigmas.append(sigma)
            a = np.random.normal(mu, sigma)
            return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        self.memory.append((s, a, r, s_))

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

        if not self.stop_signal and self.n_agent == 1:
            print("Total R:", R)
            self.agent.save(R)
            brain.add_reward(R, self.n_agent)
            if len(self.agent.rewards) % 10 == 0:
                disp()

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
    x = [np.mean(brain.sequential_rewards[max(i-50, 1):i])
         for i in range(2, len(brain.sequential_rewards))]
    plt.plot(x)
    plt.show(block=False)
    env_test.agent.disp()


env_test = Environment(0, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape[0]
NUM_ACTIONS = env_test.env.action_space.shape[0]
NONE_STATE = np.zeros(NUM_STATE)

brain = Brain()  # brain is global in A3C

envs = [Environment(i+1) for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

for o in opts:
    o.start()

for e in envs:
    e.start()

try:
    time.sleep(RUN_TIME)
except KeyboardInterrupt:
    print("End of the training")

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
    disp()
