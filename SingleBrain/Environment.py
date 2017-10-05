
import gym
import parameters


class Environment:

    def __init__(self, render=False):

        self.env = gym.make(parameters.ENV)
        self.render = render

    def get_state_size(self):
        try:
            return (self.env.observation_space.n, )
        except AttributeError:
            return list(self.env.observation_space.shape)

    def get_action_size(self):
        if parameters.ENV == "SpaceInvaders-v0":
            return 4
        return self.env.action_space.n

    def reset(self):
        return self.env.reset()

    def next_state(self, action):
        assert self.env.action_space.contains(action)
        if self.render:
            self.env.render()
        return self.env.step(action)

    def close(self):
        self.env.render(close=True)
        self.env.close()
