
import os
import gym
# from osim.env import RunEnv


class Environment:

    def __init__(self):
        # self.env = RunEnv(visualize=False)
        self.env = gym.make("Pendulum-v0")
        print()
        self.render = False

    def set_render(self, render):
        if not render:
            self.env.render(close=True)
        # if render != self.render:
        #     self.env = RunEnv(visualize=render)
        #     self.env.reset(difficulty=0)
        self.render = render

    def reset(self):
        return self.env.reset() #difficulty=0)

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
