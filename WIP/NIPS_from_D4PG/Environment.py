
import os
import numpy as np
from settings import DISPLAY
from osim.env import RunEnv

class Environment:

    def __init__(self):
        self.env = RunEnv(visualize=False)
        print()
        self.render = False

    def get_state_size(self):
        return list(self.env.observation_space.shape)

    def get_action_size(self):
        return self.env.action_space.shape[0]

    def get_bounds(self):
        return self.env.action_space.low, self.env.action_space.high

    def set_render(self, render):
        visu = render and DISPLAY
        if visu != self.render:
            self.render = visu
            self.env = RunEnv(visualize=visu)
            self.reset()

    def reset(self):
        return np.asarray(self.env.reset(difficulty=0))

    def random(self):
        return self.env.action_space.sample()

    def act(self, action):
        s_, r, d, i = self.env.step(action)
        return np.asarray(s_), r, d, i

    def close(self):
        self.env.close()
