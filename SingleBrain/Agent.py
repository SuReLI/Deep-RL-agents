
import threading
import random
import numpy as np
from time import sleep

from parameters import EPSILON_START, EPSILON_STEPS, EPSILON_STOP
from parameters import GAMMA, GAMMA_N, N_STEP_RETURN, THREAD_DELAY

from Environment import Environment
from Displayer import DISPLAYER


class Agent(threading.Thread):

    epsilon = EPSILON_START
    epsilon_decay = (EPSILON_START - EPSILON_STOP) / EPSILON_STEPS

    def __init__(self, n_agent, brain, render=False):
        print("Initializing worker n", n_agent)

        threading.Thread.__init__(self)
        self.stop_signal = False
        self.n_agent = n_agent

        self.brain = brain
        self.env = Environment(render)

        self.state_size = self.env.get_state_size()
        self.action_size = self.env.get_action_size()

        self.memory = []    # used for n_step return
        self.R = 0.

    def act(self, state):
        if Agent.epsilon > EPSILON_STOP:
            Agent.epsilon -= Agent.epsilon_decay

        if random.random() < Agent.epsilon:
            return random.randint(0, self.action_size - 1)

        else:
            state = np.array([state])
            pi, value = self.brain.predict(state)
            pi = pi[0]
            action = np.random.choice(self.action_size, p=pi)

            return action

    def get_sample(self, n):
        s, a, _, _ = self.memory[0]
        _, _, _, s_ = self.memory[n - 1]
        return s, a, self.R, s_

    def train(self, s, a, r, s_):
        # turn action into one-hot representation
        a_onehot = np.zeros(self.action_size)
        a_onehot[a] = 1

        self.memory.append((s, a_onehot, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = self.get_sample(n)
                self.brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = self.get_sample(N_STEP_RETURN)
            self.brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

    def run(self, n_episodes=None):
        step = 0

        while (n_episodes is None or step < n_episodes) \
                and not self.stop_signal:
            s = self.env.reset()

            R = 0
            done = False
            while not done and not self.stop_signal:
                sleep(THREAD_DELAY)  # yield

                a = self.act(s)
                s_, r, done, info = self.env.next_state(a)

                if done:  # terminal state
                    s_ = None

                self.train(s, a, r, s_)

                s = s_
                R += r

            if not self.stop_signal:
                DISPLAYER.add_reward(R, self.n_agent)
            step += 1
            if step % 20 == 0 and self.n_agent == 1:
                print("Reward :", R)
            if step % 200 == 0 and self.n_agent == 1:
                DISPLAYER.disp_one()

    def stop(self):
        self.stop_signal = True
        self.env.close()
