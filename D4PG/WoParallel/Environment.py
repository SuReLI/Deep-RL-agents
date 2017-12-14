
import os
import gym

class Environment:

    def __init__(self):
        self.env = gym.make("Pendulum-v0")
        print()
        self.render = False

    def get_state_size(self):
        return list(self.env.observation_space.shape)

    def get_action_size(self):
        return self.env.action_space.shape[0]

    def get_bounds(self):
        return self.env.action_space.low, self.env.action_space.high

    def set_render(self, render):
        self.render = render

    def reset(self):
        return self.env.reset()

    def random(self):
        return self.env.action_space.sample()

    def act(self, action):
        assert self.env.action_space.contains(action)
        if self.render:
            self.env.render()
        return self.env.step(action)

    def close(self):
        self.env.render(close=True)
        self.env.close()
