
import os
import gym
from osim.env import RunEnv
from parameters import ENV, FRAME_SKIP


class Environment:

    def __init__(self):

        print("Setting env...")
        self.env = RunEnv(visualize=True)
        print("Env set !")

    def get_state_size(self):
        return list(self.env.observation_space.shape)

    def get_action_size(self):
        return self.env.action_space.shape[0]

    def get_bounds(self):
        return self.env.action_space.low, self.env.action_space.high

    def reset(self):
        return self.env.reset(difficulty=0)

    def random(self):
        return self.env.action_space.sample()

    def act(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()
