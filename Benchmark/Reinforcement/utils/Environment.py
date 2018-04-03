
import os
import gym
import numpy as np
from collections import deque

from settings import Settings


class Environment:
    """
    Gym-environment wrapper to add the possibility to save GIFs.

    If the boolean gif if True, then every time the action methods is called,
    the environment keeps a picture of the environment in a list until the
    method save_gif is called.
    """

    def __init__(self):

        self.env = gym.make(Settings.ENV)

        self.frame_buffer = deque(maxlen=4)

    def set_render(self, render):
        pass

    def set_gif(self, gif, name=None):
        pass

    def reset(self):
        return self.env.reset()

    def act_random(self):
        """
        Wrapper method to return a random action.
        """
        return self.env.action_space.sample()

    def act(self, action):
        """
        Wrapper method to add frame skip.
        """
        r, i, done = 0, 0, False
        while i < (Settings.FRAME_SKIP + 1) and not done:
            s_, r_tmp, done, info = self.env.step(action)
            r += r_tmp
            i += 1

        return s_, r, done, info

    def close(self):
        """
        Close the environment and save gif under the name 'last_gif'
        """
        self.env.close()
