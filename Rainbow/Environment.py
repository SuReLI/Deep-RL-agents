
import gym
from parameters import ENV


class Environment:

    def __init__(self):

        self.env = gym.make(ENV)
        print()
        self.render = False
        self.offset = 0

    def get_state_size(self):
        try:
            return (self.env.observation_space.n, )
        except AttributeError:
            return list(self.env.observation_space.shape)

    def get_action_size(self):
        if ENV == "SpaceInvaders-v0" or ENV == "SpaceInvaders-ram-v0":
            return 4
        elif ENV == "Pong-v0" or ENV == "Pong-ram-v0":
            self.offset = 2
            return 2
        else:
            return self.env.action_space.n

    def set_render(self, render):
        self.render = render

    def reset(self):
        return self.env.reset()

    def act(self, action):
        action += self.offset
        assert self.env.action_space.contains(action)
        if self.render:
            self.env.render()
        return self.env.step(action)

    def close(self):
        self.env.render(close=True)
        self.env.close()
